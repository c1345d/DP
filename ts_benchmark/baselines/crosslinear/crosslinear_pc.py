"""CrossLinear PC adapter."""
from .crosslinear import CrossLinear, MODEL_HYPER_PARAMS
from .model.crosslinear_model_pc import CrossLinearModel_PC


class CrossLinear_PC(CrossLinear):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct

    @property
    def model_name(self):
        return "CrossLinear_PC"

    def _init_model(self):
        return CrossLinearModel_PC(self.config, correction_type=self.config.correction_type)
