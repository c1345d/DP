"""TimeKAN PC model."""
import torch
import torch.nn as nn
from .timekan_model import TimeKANModeL
from ts_benchmark.baselines.time_series_library.models.correction import apply_correction


class TimeKANModeL_PC(TimeKANModeL):
    def __init__(self, configs, correction_type='linear'):
        super().__init__(configs)
        self.correction_type = correction_type
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k): self.correction_kwargs[k] = getattr(configs, k)
        # Extend predict_layer to output seq_len+pred_len
        out_len = configs.seq_len + configs.pred_len
        self.predict_layer = nn.Linear(configs.seq_len, out_len)

    def forecast(self, x_enc):
        B = x_enc.shape[0]
        out_len = self.seq_len + self.pred_len

        # Multi-level processing (same as parent, via name-mangled method)
        x_enc_list = self._TimeKANModeL__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            B2, T, N = x.size()
            x = self.normalize_layers[i](x, "norm")
            x = x.permute(0, 2, 1).contiguous().reshape(B2 * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        # Predict with extended predict_layer
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = (
            self.projection_layer(dec_out)
            .reshape(B, self.configs.c_out, out_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        dec_out = self.normalize_layers[0](dec_out, "denorm")

        x_recon = dec_out[:, :self.seq_len, :]
        y_pred = dec_out[:, -self.pred_len:, :]
        return apply_correction(x_enc, x_recon, y_pred, self.correction_type, **self.correction_kwargs)
