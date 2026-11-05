"""
FEDformer with Predict-then-Correct framework.

Adds a reconstruction head from encoder output, then applies
correction (fused/multiband/gated).

Model name for CLI: time_series_library.FEDformer_PC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FEDformer import FEDformer
from .correction import apply_correction


class FEDformer_PC(FEDformer):
    def __init__(self, configs, version="fourier", mode_select="random", modes=32,
                 correction_type=None):
        super(FEDformer_PC, self).__init__(configs, version, mode_select, modes)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.recon_head = nn.Linear(configs.d_model, configs.c_out)

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len)
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Reconstruction from encoder output
        x_recon = self.recon_head(enc_out)  # (B, seq_len, c_out)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        dec_out = trend_part + seasonal_part
        y_pred = dec_out[:, -self.pred_len:, :]
        return self._apply_correction(x_enc, x_recon, y_pred)
