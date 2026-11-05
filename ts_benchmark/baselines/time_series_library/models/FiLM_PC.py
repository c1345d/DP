"""
FiLM with Predict-then-Correct framework.

Extends Legendre polynomial output length from pred_len to seq_len+pred_len.
"""

import torch
import torch.nn as nn

from .FiLM import FiLM
from .correction import apply_correction


class FiLM_PC(FiLM):
    def __init__(self, configs, correction_type=None):
        super(FiLM_PC, self).__init__(configs)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        # Extend the final MLP: maps stacked Legendre coeffs -> (seq_len+pred_len) instead of 1
        out_ratio = (configs.seq_len + self.pred_len) // self.pred_len + 1
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), out_ratio)

    def _forecast_extended(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec):
        """Modified forecast that outputs seq_len+pred_len instead of just pred_len."""
        out_len = self.seq_len + self.pred_len

        # Normalization (same as base)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[
                :, :, :, jump_dist:
            ]
            out1 = self.spec_conv_1[i](x_in_c)
            x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            # key change: reconstruct out_len steps instead of pred_len
            eval_mat = legt.eval_matrix
            if eval_mat.shape[0] >= out_len:
                x_dec = x_dec_c @ eval_mat[-out_len:, :].T
            else:
                x_dec = x_dec_c @ eval_mat.T
                # pad with zeros if eval_matrix is too short
                pad = torch.zeros(x_dec.shape[0], x_dec.shape[1], out_len - x_dec.shape[2],
                                  device=x_dec.device)
                x_dec = torch.cat([pad, x_dec], dim=2)
            x_decs.append(x_dec)

        x_dec = torch.stack(x_decs, dim=-1)  # (B, out_len, C, num_features)
        x_dec = self.mlp(x_dec).mean(-1).permute(0, 2, 1)  # (B, C, out_len) -> (B, out_len, C)

        # De-Normalization
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means
        return x_dec

    def forecast(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec):
        out = self._forecast_extended(x_enc, x_mark_enc, x_dec_true, x_mark_dec)
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
