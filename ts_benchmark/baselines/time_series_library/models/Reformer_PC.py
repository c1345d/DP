"""
Reformer with Predict-then-Correct framework.

Pattern B: adds recon_head from encoder output.
Reformer pads input with placeholder tokens, then projects from encoder state.
"""

import torch
import torch.nn as nn

from .Reformer import Reformer
from .correction import apply_correction


class Reformer_PC(Reformer):
    def __init__(self, configs, correction_type=None, bucket_size=4, n_hashes=4):
        super(Reformer_PC, self).__init__(configs, bucket_size=bucket_size, n_hashes=n_hashes)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        if self.task_name in ("long_term_forecast", "short_term_forecast", "imputation", "anomaly_detection"):
            self.recon_head = nn.Linear(configs.d_model, configs.c_out)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Pad with placeholder (same as base)
        x_enc_padded = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc_padded = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1
            )
        else:
            x_mark_enc_padded = None

        enc_out = self.enc_embedding(x_enc_padded, x_mark_enc_padded)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Reconstruction from encoder state (use only the original seq_len portion)
        x_recon = self.recon_head(enc_out[:, :self.seq_len, :])  # (B, seq_len, c_out)

        # Prediction
        dec_out = self.projection(enc_out)
        y_pred = dec_out[:, -self.pred_len:, :]

        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization (same as base)
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        # Pad with placeholder
        x_enc_padded = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc_padded = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1
            )
        else:
            x_mark_enc_padded = None

        enc_out = self.enc_embedding(x_enc_padded, x_mark_enc_padded)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        x_recon = self.recon_head(enc_out[:, :self.seq_len, :])
        dec_out = self.projection(enc_out)
        y_pred = dec_out[:, -self.pred_len:, :]

        y_pred = y_pred * std_enc + mean_enc
        x_recon = x_recon * std_enc + mean_enc

        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast":
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "short_term_forecast":
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
