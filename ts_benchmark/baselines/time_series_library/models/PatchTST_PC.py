"""
PatchTST with Predict-then-Correct framework.

Extends PatchTST to output seq_len+pred_len timesteps via an
extended FlattenHead, then applies correction (linear or spectral).

Model name for CLI: time_series_library.PatchTST_PC
"""

import torch
import torch.nn as nn

from .PatchTST import PatchTST, FlattenHead
from .correction import apply_correction


class PatchTST_PC(PatchTST):
    """
    PatchTST with Predict-then-Correct.

    Key changes from base PatchTST:
    - FlattenHead outputs seq_len + pred_len (instead of just pred_len)
    - forecast() splits output, applies correction, returns corrected prediction
    - forward() unchanged (inherited) - output is still (B, pred_len, C)

    Config param (passed via model-hyper-params):
    - correction_type: 'linear' (default), 'spectral', or 'none'
    """

    def __init__(self, configs, correction_type=None):
        super(PatchTST_PC, self).__init__(configs)

        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        out_len = configs.seq_len + configs.pred_len

        # Override head to output seq_len+pred_len
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                out_len,                          # was configs.pred_len
                head_dropout=configs.dropout,
            )
        elif self.task_name == "imputation" or self.task_name == "anomaly_detection":
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                out_len,
                head_dropout=configs.dropout,
            )

    def _apply_correction(self, x_true, x_recon, y_pred):
        """Apply correction to prediction."""
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Predict seq_len+pred_len via encoder + extended head, then correct.
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc_norm = x_enc_norm / stdev

        # Patching and embedding
        x_enc_norm = x_enc_norm.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc_norm)

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder (extended head: seq_len + pred_len)
        dec_out = self.head(enc_out)                     # (B, n_vars, seq_len+pred_len)
        dec_out = dec_out.permute(0, 2, 1)               # (B, seq_len+pred_len, C)

        # De-Normalization
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, dec_out.shape[1], 1)
        )

        # Split into history reconstruction and future prediction
        x_recon = dec_out[:, :self.seq_len, :]           # (B, seq_len, C)
        y_pred = dec_out[:, -self.pred_len:, :]          # (B, pred_len, C)

        return self._apply_correction(x_enc, x_recon, y_pred)
