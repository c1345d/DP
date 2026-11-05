"""PatchMLP PC adapter."""
from .patchmlp import PatchMLP, MODEL_HYPER_PARAMS
from .models.patchmlp_model_pc import PatchMLPModel_PC


class PatchMLP_PC(PatchMLP):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct
        for k in ['tau', 'delta', 'lam', 'n_bands']:
            if k in kwargs: setattr(self.config, k, kwargs[k])

    @property
    def model_name(self):
        return "PatchMLP_PC"

    def _init_model(self):
        return PatchMLPModel_PC(self.config, correction_type=self.config.correction_type)
