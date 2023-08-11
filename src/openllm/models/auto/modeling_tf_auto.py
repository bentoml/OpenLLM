from __future__ import annotations
from collections import OrderedDict
from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import BaseAutoLLMClass
from .factory import _LazyAutoMapping

MODEL_TF_MAPPING_NAMES = OrderedDict([("flan_t5", "TFFlanT5"), ("opt", "TFOPT")])
MODEL_TF_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_TF_MAPPING_NAMES)
class AutoTFLLM(BaseAutoLLMClass):
  _model_mapping = MODEL_TF_MAPPING
