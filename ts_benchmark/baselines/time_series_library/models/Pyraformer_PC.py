"""
Pyraformer with Predict-then-Correct framework.

Pattern A: extends projection layer from pred_len*enc_in to (seq_len+pred_len)*enc_in.
"""

import torch
import torch.nn as nn

from .Pyraformer import Pyraformer
from .correction import apply_correction


class Pyraformer_PC(Pyraformer):
    def __init__(self, configs, correction_type=None, window_size=[4, 4], inner_size=5):
        super(Pyraformer_PC, self).__init__(configs, window_size=window_size, inner_size=inner_size)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        # Extend projection: pred_len * enc_in -> (seq_len+pred_len) * enc_in
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.out_len = configs.seq_len + configs.pred_len
            self.projection = nn.Linear(
                (len(window_size) + 1) * self.d_model, self.out_len * configs.enc_in
            )

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(enc_out.size(0), self.out_len, -1)
        x_recon = dec_out[:, :self.seq_len, :]
        y_pred = dec_out[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization (same as base)
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_out = self.projection(enc_out).view(enc_out.size(0), self.out_len, -1)
        dec_out = dec_out * std_enc + mean_enc

        x_recon = dec_out[:, :self.seq_len, :]
        y_pred = dec_out[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast":
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "short_term_forecast":
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
