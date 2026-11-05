"""FITS with Predict-then-Correct."""
import torch
import torch.nn as nn
from .fits_model import FITSModel
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class FITSModel_PC(FITSModel):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k): self.correction_kwargs[k] = getattr(configs, k)

    def forward(self, x):
        out, low = FITSModel.forward(self, x)  # base returns (xy, low_xy), both (B, seq_len+pred_len, C)
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        corrected = apply_correction(x, x_recon, y_pred, self.correction_type, **self.correction_kwargs)
        return corrected, low  # keep tuple to match adapter's _process unpacking
