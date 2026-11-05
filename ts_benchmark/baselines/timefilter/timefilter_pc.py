"""TimeFilter PC adapter."""
from .timefilter import TimeFilter, MODEL_HYPER_PARAMS
from .models.TimeFilter_pc import Model_PC


class TimeFilter_PC(TimeFilter):
    def __init__(self, correction_type=None, **kwargs):
        super().__init__(**kwargs)
        ct = correction_type or getattr(self.config, 'correction_type', 'linear')
        self.config.correction_type = ct

    @property
    def model_name(self):
        return "TimeFilter_PC"

    def _init_model(self):
        return Model_PC(self.config, correction_type=self.config.correction_type)
