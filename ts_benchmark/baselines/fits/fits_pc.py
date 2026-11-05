"""FITS PC adapter."""
from .fits import FITS, MODEL_HYPER_PARAMS
from .fits_model_pc import FITSModel_PC


class FITS_PC(FITS):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if k in kwargs: setattr(self.config, k, kwargs[k])

    @property
    def model_name(self):
        return "FITS_PC"

    def _init_model(self):
        return FITSModel_PC(self.config, correction_type=self.config.correction_type)
