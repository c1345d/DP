"""
Koopa with Predict-then-Correct framework.

Pattern A: extends forecast output from pred_len to seq_len+pred_len.
Koopa uses Koopman operator dynamics; we modify the time-invariant decoder output dimension.
"""

import torch
import torch.nn as nn

from .Koopa import Koopa, MLP
from .correction import apply_correction


class Koopa_PC(Koopa):
    def __init__(self, configs, correction_type=None, dynamic_dim=128, hidden_dim=64,
                 hidden_layers=2, num_blocks=3, multistep=False):
        super(Koopa_PC, self).__init__(
            configs,
            dynamic_dim=dynamic_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            num_blocks=num_blocks,
            multistep=multistep,
        )
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        # Extend time-invariant decoder: pred_len -> seq_len + pred_len
        self.out_len = configs.seq_len + configs.pred_len
        self.time_inv_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.out_len,  # was pred_len
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # Re-build time-invariant KP blocks with extended decoder
        self.time_inv_kps = nn.ModuleList([
            type(self.time_inv_kps[0]).__class__(
                input_len=self.input_len,
                pred_len=self.out_len,  # extended
                dynamic_dim=self.dynamic_dim,
                encoder=self.time_inv_encoder,
                decoder=self.time_inv_decoder,
            )
            for _ in range(self.num_blocks)
        ])

    def forecast(self, x_enc):
        # Series Stationarization (same as base)
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting (same as base, but time_inv_kps now output out_len)
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)  # (B, out_len, C)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = time_inv_output + time_var_output
            else:
                forecast += time_inv_output + time_var_output

        # Denorm
        res = forecast * std_enc + mean_enc

        # Split and apply correction
        x_recon = res[:, :self.input_len, :]
        y_pred = res[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == "long_term_forecast":
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec)
