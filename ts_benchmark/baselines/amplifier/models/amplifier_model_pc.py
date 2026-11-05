"""Amplifier PC model."""
import torch
import torch.nn as nn
from .amplifier_model import AmplifierModel
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class AmplifierModel_PC(AmplifierModel):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k): self.correction_kwargs[k] = getattr(configs, k)
        out_len = configs.seq_len + configs.pred_len
        # Extend all three output heads
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, out_len))
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, out_len))
        self.freq_linear = nn.Linear(
            int(self.seq_len / 2) + 1, int(out_len / 2) + 1).to(torch.cfloat)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = AmplifierModel.forward(self, x, x_mark_enc, x_dec, x_mark_dec)  # seq_len+pred_len
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        return apply_correction(x, x_recon, y_pred, self.correction_type, **self.correction_kwargs)
