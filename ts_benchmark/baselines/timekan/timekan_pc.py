"""TimeKAN PC adapter."""
from .timekan import TimeKAN, MODEL_HYPER_PARAMS
from .models.timekan_model_pc import TimeKANModeL_PC


class TimeKAN_PC(TimeKAN):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct

    @property
    def model_name(self):
        return "TimeKAN_PC"

    def _init_model(self):
        return TimeKANModeL_PC(self.config, correction_type=self.config.correction_type)
