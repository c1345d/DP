"""
MICN with Predict-then-Correct framework.

MICN's decoder already processes seq_len+pred_len internally.
We extend the trend regression to seq_len+pred_len and remove
the premature truncation, then split and apply correction.

Model name for CLI: time_series_library.MICN_PC
"""

import torch
import torch.nn as nn

from .MICN import MICN
from .correction import apply_correction


class MICN_PC(MICN):
    def __init__(self, configs, conv_kernel=[12, 16], correction_type=None):
        super(MICN_PC, self).__init__(configs, conv_kernel)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        # Extend trend regression to output seq_len+pred_len
        out_len = configs.seq_len + configs.pred_len
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.regression = nn.Linear(configs.seq_len, out_len)
            self.regression.weight = nn.Parameter(
                (1 / out_len) * torch.ones([out_len, configs.seq_len]),
                requires_grad=True,
            )

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        # Trend: seq_len -> seq_len+pred_len
        trend_full = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        # (B, seq_len+pred_len, C)

        # embedding - already seq_len+pred_len
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init_dec = torch.cat(
            [seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1
        )
        dec_out = self.dec_embedding(seasonal_init_dec, x_mark_dec)
        dec_out = self.conv_trans(dec_out)  # (B, seq_len+pred_len, C)
        dec_out = dec_out + trend_full       # (B, seq_len+pred_len, C)

        # Split: reconstruction + prediction
        x_recon = dec_out[:, :self.seq_len, :]        # (B, seq_len, C)
        y_pred = dec_out[:, -self.pred_len:, :]       # (B, pred_len, C)

        return self._apply_correction(x_enc, x_recon, y_pred)
