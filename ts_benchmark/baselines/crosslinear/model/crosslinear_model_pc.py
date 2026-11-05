"""CrossLinear PC model."""
import torch
import torch.nn as nn
from .crosslinear_model import CrossLinearModel
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class CrossLinearModel_PC(CrossLinearModel):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k): self.correction_kwargs[k] = getattr(configs, k)
        # Extend De_Patch_Embedding output
        out_len = configs.seq_len + configs.pred_len
        from .crosslinear_model import De_Patch_Embedding
        d_model = configs.d_model
        d_ff = configs.d_ff
        variate_num = configs.dec_in  # self.ms is False, so variate_num = dec_in
        patch_len = getattr(configs, 'patch_len', 16)
        patch_num = math.ceil(configs.seq_len / patch_len)
        self.de_patch_embedding = De_Patch_Embedding(out_len, patch_num, d_model, d_ff, variate_num)

    def forward(self, x_enc):
        out = CrossLinearModel.forward(self, x_enc)  # seq_len+pred_len
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)
