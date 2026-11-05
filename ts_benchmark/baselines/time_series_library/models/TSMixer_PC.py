"""
TSMixer with Predict-then-Correct framework.

TSMixer uses a single Linear(seq_len, pred_len) projection head.
PC variant extends this to seq_len+pred_len, then splits and corrects.

Model name for CLI: time_series_library.TSMixer_PC
"""

import torch
import torch.nn as nn

from .TSMixer import TSMixer
from .correction import apply_correction


class TSMixer_PC(TSMixer):
    def __init__(self, configs, correction_type=None):
        super(TSMixer_PC, self).__init__(configs)
        self.seq_len = configs.seq_len   # parent doesn't store this
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        out_len = configs.seq_len + configs.pred_len
        self.projection = nn.Linear(configs.seq_len, out_len)

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)  # (B, seq_len+pred_len, C)

        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        return self._apply_correction(x_enc, x_recon, y_pred)
