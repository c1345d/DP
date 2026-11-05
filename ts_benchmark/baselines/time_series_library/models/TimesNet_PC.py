"""
TimesNet with Predict-then-Correct framework.

TimesNet already outputs seq_len+pred_len in its forecast() via
predict_linear. The PC variant simply splits this output into
history reconstruction + future prediction and applies correction.

Model name for CLI: time_series_library.TimesNet_PC
"""

import torch

from .TimesNet import TimesNet
from .correction import apply_correction


class TimesNet_PC(TimesNet):
    """
    TimesNet with Predict-then-Correct.

    No head modification needed - TimesNet's forecast() already produces
    seq_len+pred_len outputs. We only override forecast() to split and correct.
    """

    def __init__(self, configs, correction_type=None):
        super(TimesNet_PC, self).__init__(configs)
        self.correction_type = correction_type or getattr(configs, 'correction_type', 'linear')
        self.correction_kwargs = {}
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if hasattr(configs, k):
                self.correction_kwargs[k] = getattr(configs, k)

    def _apply_correction(self, x_true, x_recon, y_pred):
        return apply_correction(x_true, x_recon, y_pred, self.correction_type, **self.correction_kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Get full seq_len+pred_len output from parent forecast
        out = TimesNet.forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec)

        # Split into history reconstruction and future prediction
        x_recon = out[:, :self.seq_len, :]
        y_pred = out[:, -self.pred_len:, :]

        return self._apply_correction(x_enc, x_recon, y_pred)
