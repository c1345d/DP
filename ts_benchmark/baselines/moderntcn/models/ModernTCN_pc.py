"""ModernTCN PC model."""
import torch
import torch.nn as nn
from .ModernTCN import Model, Flatten_Head
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class Model_PC(Model):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k): self.correction_kwargs[k] = getattr(configs, k)

        out_len = configs.seq_len + configs.pred_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 2)
        d_model = configs.dims[-1]
        n_vars = configs.enc_in
        head_nf = d_model * patch_num
        head_drop = getattr(configs, 'head_dropout', 0.1)

        # Replace heads on inner models
        if self.decomposition:
            self.model_res.head = Flatten_Head(self.individual, n_vars, head_nf, out_len, head_dropout=head_drop)
            self.model_trend.head = Flatten_Head(self.individual, n_vars, head_nf, out_len, head_dropout=head_drop)
        else:
            self.model.head = Flatten_Head(self.individual, n_vars, head_nf, out_len, head_dropout=head_drop)

    def forward(self, x, te=None):
        out = Model.forward(self, x, te)   # now seq_len+pred_len via extended head
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        return apply_correction(x, x_recon, y_pred, self.correction_type, **self.correction_kwargs)
