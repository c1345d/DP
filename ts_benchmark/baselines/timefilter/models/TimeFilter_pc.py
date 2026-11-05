"""TimeFilter PC model."""
import torch
import torch.nn as nn
from .TimeFilter import Model
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class Model_PC(Model):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)
        # Extend head: pred_len -> seq_len + pred_len
        out_len = configs.seq_len + configs.pred_len
        self.head = nn.Linear(self.dim * self.num_patches, out_len)

    def forward(self, x, masks, is_training=False, target=None):
        # x: [B, T, C]
        B, T, C = x.shape
        # Normalization
        x = self.norm(x, 'norm')
        # x: [B, C, T]
        x = x.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        x = self.patch_embed(x)  # [B, N, D]  N = [C*T / P]
        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)

        # [B, C, T/P, D] -> head -> [B, C, out_len]
        out_len = self.seq_len + self.pred_len
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))  # [B, C, out_len]
        x = x.permute(0, 2, 1)  # [B, out_len, C]
        # De-Normalization
        x = self.norm(x, 'denorm')

        # Split and apply correction
        x_enc = x[:, :self.seq_len, :]
        x_recon = x[:, :self.seq_len, :]
        y_pred = x[:, -self.pred_len:, :]
        corrected = apply_correction(
            x_enc, x_recon, y_pred,
            self.correction_type, **self.correction_kwargs
        )

        return corrected, moe_loss
