"""SparseTSF PC adapter."""
from .sparsetsf import SparseTSF, MODEL_HYPER_PARAMS
from .models.sparsetsf_model_pc import SparseTSFMoedl_PC


class SparseTSF_PC(SparseTSF):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct

    @property
    def model_name(self):
        return "SparseTSF_PC"

    def _init_model(self):
        return SparseTSFMoedl_PC(self.config, correction_type=self.config.correction_type)
