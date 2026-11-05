"""PatchMLP PC model."""
import torch
import torch.nn as nn
from .patchmlp_model import PatchMLPModel
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class PatchMLPModel_PC(PatchMLPModel):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k): self.correction_kwargs[k] = getattr(configs, k)
        self.projector = nn.Linear(configs.d_model, configs.seq_len + configs.pred_len, bias=True)

    def forecast(self, x_enc):
        out_len = self.seq_len + self.pred_len
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)

        seasonal_init, trend_init = self.decompsition(x)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, out_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, out_len, 1)
            )

        x_recon = dec_out[:, :self.seq_len, :]
        y_pred = dec_out[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)
