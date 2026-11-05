"""
Crossformer with Predict-then-Correct framework.

Adds a reconstruction head from encoder output (using FlattenHead
like the imputation path), then applies correction.

Model name for CLI: time_series_library.Crossformer_PC
"""

import torch
import torch.nn as nn

from einops import rearrange, repeat
from .Crossformer import Crossformer
from .PatchTST import FlattenHead
from .correction import apply_correction


class Crossformer_PC(Crossformer):
    def __init__(self, configs, correction_type=None):
        super(Crossformer_PC, self).__init__(configs)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        # Reconstruction head (same pattern as imputation head)
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.recon_head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
            )

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc_emb, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc_emb = rearrange(
            x_enc_emb, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc_emb += self.enc_pos_embedding
        x_enc_emb = self.pre_norm(x_enc_emb)
        enc_out, attns = self.encoder(x_enc_emb)

        # Reconstruction from encoder output (last scale)
        x_recon = self.recon_head(
            enc_out[-1].permute(0, 1, 3, 2)
        ).permute(0, 2, 1)  # (B, seq_len, C)

        # dec
        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_enc_emb.shape[0],
        )
        dec_out = self.decoder(dec_in, enc_out)
        y_pred = dec_out[:, -self.pred_len:, :]

        return self._apply_correction(x_enc, x_recon, y_pred)
