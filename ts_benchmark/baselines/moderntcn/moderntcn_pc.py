"""ModernTCN PC adapter."""
from .moderntcn import ModernTCN, MODEL_HYPER_PARAMS
from .models.ModernTCN_pc import Model_PC


class ModernTCN_PC(ModernTCN):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct

    @property
    def model_name(self):
        return "ModernTCN_PC"

    def _init_model(self):
        return Model_PC(self.config, correction_type=self.config.correction_type)
