# -*- coding: utf-8 -*-
__all__ = [
    "Autoformer",
    "Autoformer_PC",
    "Crossformer",
    "Crossformer_PC",
    "DLinear",
    "DLinear_PC",
    "ETSformer",
    "FEDformer",
    "FEDformer_PC",
    "FiLM",
    "FreTS",
    "FreTS_PC",
    "Informer",
    "Informer_PC",
    "iTransformer",
    "iTransformer_PC",
    "Koopa",
    "LightTS",
    "LightTS_PC",
    "Linear",
    "MICN",
    "MICN_PC",
    "NLinear",
    "NLinear_PC",
    "Nonstationary_Transformer",
    "PatchTST",
    "PatchTST_PC",
    "Pyraformer",
    "Reformer",
    "TimesNet",
    "TimesNet_PC",
    "Transformer",
    "Transformer_PC",
    "Triformer",
    "TimeMixer",
    "TSMixer",
    "TSMixer_PC",
    "FiLM_PC",
    "Nonstationary_Transformer_PC",
    "Reformer_PC",
    "Pyraformer_PC",
    "Koopa_PC",
]


from ts_benchmark.baselines.time_series_library.models.Autoformer import Autoformer
from ts_benchmark.baselines.time_series_library.models.Crossformer import Crossformer
from ts_benchmark.baselines.time_series_library.models.DLinear import DLinear
from ts_benchmark.baselines.time_series_library.models.ETSformer import ETSformer
from ts_benchmark.baselines.time_series_library.models.FEDformer import FEDformer
from ts_benchmark.baselines.time_series_library.models.FiLM import FiLM
from ts_benchmark.baselines.time_series_library.models.FreTS import FreTS
from ts_benchmark.baselines.time_series_library.models.Informer import Informer
from ts_benchmark.baselines.time_series_library.models.iTransformer import iTransformer
from ts_benchmark.baselines.time_series_library.models.Koopa import Koopa
from ts_benchmark.baselines.time_series_library.models.LightTS import LightTS
from ts_benchmark.baselines.time_series_library.patchs.Linear import Linear
from ts_benchmark.baselines.time_series_library.models.MICN import MICN
from ts_benchmark.baselines.time_series_library.patchs.NLinear import NLinear
from ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer import (
    Nonstationary_Transformer,
)
from ts_benchmark.baselines.time_series_library.models.PatchTST import PatchTST
from ts_benchmark.baselines.time_series_library.models.Pyraformer import Pyraformer
from ts_benchmark.baselines.time_series_library.models.Reformer import Reformer
from ts_benchmark.baselines.time_series_library.models.TimesNet import TimesNet
from ts_benchmark.baselines.time_series_library.models.Transformer import Transformer
from ts_benchmark.baselines.time_series_library.patchs.Triformer import Triformer
from ts_benchmark.baselines.time_series_library.models.TimeMixer import TimeMixer
from ts_benchmark.baselines.time_series_library.models.TSMixer import TSMixer

# Predict-then-Correct variants (safe import - won't crash if file missing)
import logging
_logger = logging.getLogger(__name__)

_PC_IMPORTS = [
    ("Autoformer_PC", "ts_benchmark.baselines.time_series_library.models.Autoformer_PC"),
    ("Crossformer_PC", "ts_benchmark.baselines.time_series_library.models.Crossformer_PC"),
    ("DLinear_PC", "ts_benchmark.baselines.time_series_library.models.DLinear_PC"),
    ("FEDformer_PC", "ts_benchmark.baselines.time_series_library.models.FEDformer_PC"),
    ("FreTS_PC", "ts_benchmark.baselines.time_series_library.models.FreTS_PC"),
    ("Informer_PC", "ts_benchmark.baselines.time_series_library.models.Informer_PC"),
    ("iTransformer_PC", "ts_benchmark.baselines.time_series_library.models.iTransformer_PC"),
    ("LightTS_PC", "ts_benchmark.baselines.time_series_library.models.LightTS_PC"),
    ("MICN_PC", "ts_benchmark.baselines.time_series_library.models.MICN_PC"),
    ("PatchTST_PC", "ts_benchmark.baselines.time_series_library.models.PatchTST_PC"),
    ("TimesNet_PC", "ts_benchmark.baselines.time_series_library.models.TimesNet_PC"),
    ("Transformer_PC", "ts_benchmark.baselines.time_series_library.models.Transformer_PC"),
    ("TSMixer_PC", "ts_benchmark.baselines.time_series_library.models.TSMixer_PC"),
    ("NLinear_PC", "ts_benchmark.baselines.time_series_library.patchs.NLinear_PC"),
    ("FiLM_PC", "ts_benchmark.baselines.time_series_library.models.FiLM_PC"),
    ("Nonstationary_Transformer_PC", "ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer_PC"),
    ("Reformer_PC", "ts_benchmark.baselines.time_series_library.models.Reformer_PC"),
    ("Pyraformer_PC", "ts_benchmark.baselines.time_series_library.models.Pyraformer_PC"),
    ("Koopa_PC", "ts_benchmark.baselines.time_series_library.models.Koopa_PC"),
]

for _name, _path in _PC_IMPORTS:
    try:
        _pkg, _cls = _path.rsplit(".", 1)
        _mod = __import__(_path, fromlist=[_cls])
        globals()[_name] = getattr(_mod, _cls)
    except Exception as _e:
        _logger.warning("PC model '%s' not available (%s), skipping.", _name, _e)
        if _name in __all__:
            __all__.remove(_name)