"""
iTransformer with Predict-then-Correct framework.

Extends iTransformer to output seq_len+pred_len timesteps via an
extended projection head, then applies correction (linear or spectral).

Model name for CLI: time_series_library.iTransformer_PC
"""

import torch
import torch.nn as nn

from .iTransformer import iTransformer
from .correction import apply_correction, compute_correction_alpha


class iTransformer_PC(iTransformer):
    """
    iTransformer with Predict-then-Correct.

    Key changes from base iTransformer:
    - Projection head outputs seq_len + pred_len (instead of just pred_len)
    - forecast() splits output, applies correction, returns corrected prediction
    - forward() unchanged (inherited) - output is still (B, pred_len, C)

    Config param (passed via model-hyper-params):
    - correction_type: 'linear' (default), 'spectral', or 'none'
    """

    def __init__(self, configs, correction_type=None):
        super(iTransformer_PC, self).__init__(configs)

        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        out_len = configs.seq_len + configs.pred_len

        # Override projection head to output seq_len+pred_len
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.projection = nn.Linear(configs.d_model, out_len, bias=True)
        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, out_len, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, out_len, bias=True)

    def _apply_correction(self, x_true, x_recon, y_pred):
        """Apply correction to prediction."""
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def _split_raw_output(self, x_enc, x_mark_enc=None):
        """
        Predict seq_len+pred_len via encoder + extended projection, then split.
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc_norm = x_enc_norm / stdev

        _, _, N = x_enc_norm.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc_norm, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection (extended: seq_len + pred_len)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # dec_out shape: (B, seq_len+pred_len, C)

        # De-Normalization
        out_len = dec_out.shape[1]
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, out_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, out_len, 1)
        )

        # Split into history reconstruction and future prediction
        x_recon = dec_out[:, :self.seq_len, :]           # (B, seq_len, C)
        y_pred = dec_out[:, -self.pred_len:, :]          # (B, pred_len, C)

        return x_recon, y_pred

    def forecast_diagnostics(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Return PCF internals for analysis without changing forecast behavior.
        """
        x_recon, y_pred = self._split_raw_output(x_enc, x_mark_enc)
        alpha_components = compute_correction_alpha(
            x_enc,
            x_recon,
            self.correction_type,
            return_components=True,
            **self.correction_kwargs,
        )
        y_corrected = alpha_components['alpha'] * y_pred
        return {
            'x_obs': x_enc,
            'x_recon': x_recon,
            'y_raw': y_pred,
            'y_corr': y_corrected,
            'alpha_hist': alpha_components['alpha'],
            'alpha_lin_hist': alpha_components.get('alpha_lin'),
            'alpha_spec_hist': alpha_components.get('alpha_spec'),
            'agree_hist': alpha_components.get('agree'),
        }

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Predict seq_len+pred_len via encoder + extended projection, then correct.
        """
        x_recon, y_pred = self._split_raw_output(x_enc, x_mark_enc)
        return self._apply_correction(x_enc, x_recon, y_pred)
