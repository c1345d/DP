"""
FreTS with Predict-then-Correct framework.

Extends the frequency-domain MLP to output seq_len+pred_len
timesteps, then applies correction.

FreTS uses FFT + learnable frequency-domain weights. The PC variant
extends the final fc layer to predict both reconstruction and forecast.

Model name for CLI: time_series_library.FreTS_PC
"""

import torch
import torch.nn as nn

from .FreTS import FreTS
from .correction import apply_correction


class FreTS_PC(FreTS):
    """
    FreTS with Predict-then-Correct.

    Key change from base FreTS:
    - fc layer outputs seq_len+pred_len (instead of just pred_len)
    - forecast() splits output, applies correction
    """

    def __init__(self, configs, correction_type=None):
        super(FreTS_PC, self).__init__(configs)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        out_len = configs.seq_len + configs.pred_len

        # Replace fc to output seq_len+pred_len
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, out_len),
        )

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc):
        # Parent forecast with modified fc produces seq_len+pred_len
        out = FreTS.forecast(self, x_enc)             # (B, seq_len+pred_len, C)

        # Split into history reconstruction and future prediction
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]

        return self._apply_correction(x_enc, x_recon, y_pred)
