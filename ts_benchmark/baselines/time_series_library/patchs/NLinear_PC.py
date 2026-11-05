"""
NLinear with Predict-then-Correct framework.

Extends NLinear to output seq_len+pred_len timesteps, splits into
history reconstruction + future prediction, and applies correction
(linear or spectral) to the prediction.

Model name for CLI: time_series_library.NLinear_PC
"""

import torch
import torch.nn as nn

from .NLinear import NLinear
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction, compute_correction_alpha


class NLinear_PC(NLinear):
    """
    NLinear with Predict-then-Correct.

    Key changes from base NLinear:
    - Linear layer outputs seq_len + pred_len (instead of just pred_len)
    - forecast() splits output, applies correction, returns corrected prediction
    - forward() unchanged (inherited) - output is still (B, pred_len, C)

    Config param (passed via model-hyper-params):
    - correction_type: 'linear' (default), 'spectral', or 'none'
    """

    def __init__(self, configs, individual=False, correction_type=None):
        super(NLinear_PC, self).__init__(configs, individual)

        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        out_len = self.seq_len + self.pred_len

        # Replace Linear layer with extended output dimension
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, out_len))
        else:
            self.Linear = nn.Linear(self.seq_len, out_len)

    def _apply_correction(self, x_true, x_recon, y_pred):
        """Apply correction to prediction."""
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def _split_raw_output(self, x_enc):
        out = self.encoder(x_enc)
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        return x_recon, y_pred

    def forecast_diagnostics(self, x_enc):
        """
        Return PCF internals for analysis without changing forecast behavior.
        """
        x_recon, y_pred = self._split_raw_output(x_enc)
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

    def encoder(self, x):
        """
        Override encoder to use seq_len+pred_len output dimension.
        """
        out_len = self.seq_len + self.pred_len
        seq_last = x[:, -1:, :].detach()
        x_norm = x - seq_last

        if self.individual:
            output = torch.zeros(
                [x_norm.size(0), out_len, x_norm.size(2)], dtype=x_norm.dtype
            ).to(x_norm.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x_norm[:, :, i])
            x_out = output
        else:
            x_out = self.Linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        x_out = x_out + seq_last
        return x_out  # (B, out_len, C)

    def forecast(self, x_enc):
        """
        Predict seq_len+pred_len, split, apply correction.

        Args:
            x_enc: (B, seq_len, C) input window

        Returns:
            (B, pred_len, C) corrected prediction
        """
        out = self.encoder(x_enc)                       # (B, seq_len+pred_len, C)

        # Split into history reconstruction and future prediction
        x_recon = out[:, :self.seq_len, :]              # (B, seq_len, C)
        y_pred = out[:, -self.pred_len:, :]             # (B, pred_len, C)

        return self._apply_correction(x_enc, x_recon, y_pred)
