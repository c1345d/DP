"""
LightTS with Predict-then-Correct framework.

Extends the chunk-based MLP-Mixer to output seq_len+pred_len
timesteps, then applies correction.

Model name for CLI: time_series_library.LightTS_PC
"""

import torch
import torch.nn as nn

from .LightTS import LightTS
from .correction import apply_correction


class LightTS_PC(LightTS):
    """
    LightTS with Predict-then-Correct.

    Key changes from base LightTS:
    - layer_3 outputs seq_len+pred_len (instead of just pred_len)
    - AR component outputs seq_len+pred_len
    - forecast() splits output, applies correction
    """

    def __init__(self, configs, correction_type=None):
        super(LightTS_PC, self).__init__(configs)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

        out_len = configs.seq_len + configs.pred_len

        if (
            configs.task_name == "long_term_forecast"
            or configs.task_name == "short_term_forecast"
        ):
            from .LightTS import IEBlock
            # Replace layer_3 to output seq_len+pred_len
            self.layer_3 = IEBlock(
                input_dim=self.d_model // 2,
                hid_dim=self.d_model // 2,
                output_dim=out_len,
                num_node=self.enc_in,
            )
            # Replace AR to output seq_len+pred_len
            self.ar = nn.Linear(self.seq_len, out_len)

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # encoder() already returns full seq_len+pred_len through modified layer_3 + ar
        out = self.encoder(x_enc)                     # (B, seq_len+pred_len, C)

        # Split into history reconstruction and future prediction
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]

        return self._apply_correction(x_enc, x_recon, y_pred)
