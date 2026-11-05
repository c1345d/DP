"""SparseTSF PC model."""
import torch
import torch.nn as nn
from .sparsetsf_model import SparseTSFMoedl
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class SparseTSFMoedl_PC(SparseTSFMoedl):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        # Extend head: seg_num_y -> seg_num_x + seg_num_y
        out_seg = self.seg_num_x + self.seg_num_y
        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, out_seg, bias=False)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model), nn.ReLU(),
                nn.Linear(self.d_model, out_seg))

    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,seg_num_x+seg_num_y
        else:
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s  (m = seg_num_x+seg_num_y)
        out_len = self.seq_len + self.pred_len
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, out_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        # Split and apply correction
        x_recon = y[:, :self.seq_len, :]
        y_pred = y[:, -self.pred_len:, :]
        return apply_correction(
            x_recon.detach() if self.training else x_recon,
            x_recon, y_pred,
            self.correction_type, **self.correction_kwargs
        )

    def forecast(self, x_enc):
        return self.forward(x_enc)
