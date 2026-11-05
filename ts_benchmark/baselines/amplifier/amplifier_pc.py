"""Amplifier PC adapter."""
from .amplifier import Amplifier, MODEL_HYPER_PARAMS
from .models.amplifier_model_pc import AmplifierModel_PC


class Amplifier_PC(Amplifier):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct

    @property
    def model_name(self):
        return "Amplifier_PC"

    def _init_model(self):
        return AmplifierModel_PC(self.config, correction_type=self.config.correction_type)
