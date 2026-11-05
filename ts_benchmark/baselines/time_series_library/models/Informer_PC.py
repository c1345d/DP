"""
Informer with Predict-then-Correct framework.

Adds a reconstruction head from encoder output, then applies
correction (fused/multiband/gated).

Model name for CLI: time_series_library.Informer_PC
"""

import torch
import torch.nn as nn

from .Informer import Informer
from .correction import apply_correction


class Informer_PC(Informer):
    def __init__(self, configs, correction_type=None):
        super(Informer_PC, self).__init__(configs)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.recon_head = nn.Linear(configs.d_model, configs.c_out)

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_embed = self.enc_embedding(x_enc, x_mark_enc)          # (B, seq_len, d_model)
        enc_out, attns = self.encoder(enc_embed, attn_mask=None)    # (B, seq_len//2, d_model) after distillation
        # Reconstruction from pre-distillation embedding (keeps full seq_len)
        x_recon = self.recon_head(enc_embed)  # (B, seq_len, c_out)
        # dec
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        y_pred = dec_out[:, -self.pred_len:, :]
        return self._apply_correction(x_enc, x_recon, y_pred)

    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization (same as base Informer)
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc_norm = x_enc_norm / std_enc

        enc_embed = self.enc_embedding(x_enc_norm, x_mark_enc)         # (B, seq_len, d_model)
        enc_out, attns = self.encoder(enc_embed, attn_mask=None)       # (B, seq_len//2, d_model)
        # Reconstruction from pre-distillation embedding
        x_recon = self.recon_head(enc_embed)  # (B, seq_len, c_out)
        # dec
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        y_pred = dec_out[:, -self.pred_len:, :]
        return self._apply_correction(x_enc, x_recon, y_pred)
