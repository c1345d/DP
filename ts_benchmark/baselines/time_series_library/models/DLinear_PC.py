"""
DLinear with Predict-then-Correct framework.

Extends DLinear to output seq_len+pred_len timesteps, splits into
history reconstruction + future prediction, and applies correction
(linear or spectral) to the prediction.

Model name for CLI: time_series_library.DLinear_PC
"""

import torch
import torch.nn as nn

from .DLinear import DLinear
from .correction import apply_correction, compute_correction_alpha


class DLinear_PC(DLinear):
    """
    DLinear with Predict-then-Correct.

    Key changes from base DLinear:
    - Linear layers output seq_len + pred_len (instead of just pred_len)
    - forecast() splits output, applies correction, returns corrected prediction
    - forward() unchanged (inherited) - output is still (B, pred_len, C)

    Config param (passed via model-hyper-params):
    - correction_type: 'linear' (default), 'spectral', or 'none'
    """

    def __init__(self, configs, individual=False, correction_type=None):
        super(DLinear_PC, self).__init__(configs, individual)

        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        out_len = self.seq_len + self.pred_len

        # Replace Linear layers with extended output dimension
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, out_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, out_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([out_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([out_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, out_len)
            self.Linear_Trend = nn.Linear(self.seq_len, out_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([out_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([out_len, self.seq_len])
            )

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
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), out_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), out_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :]
                )
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # (B, out_len, C)

    def forecast(self, x_enc):
        """
        Predict seq_len+pred_len, split, apply correction.

        Args:
            x_enc: (B, seq_len, C) input window

        Returns:
            (B, pred_len, C) corrected prediction
        """
        x_recon, y_pred = self._split_raw_output(x_enc)

        return self._apply_correction(x_enc, x_recon, y_pred)
