from __future__ import annotations
import typing as t
from collections import OrderedDict
from .factory import BaseAutoLLMClass, _LazyAutoMapping
from openllm_core.config import CONFIG_MAPPING_NAMES
MODEL_TF_MAPPING_NAMES = OrderedDict([("flan_t5", "TFFlanT5"), ("opt", "TFOPT")])
MODEL_TF_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_TF_MAPPING_NAMES)
class AutoTFLLM(BaseAutoLLMClass):
  _model_mapping: t.ClassVar = MODEL_TF_MAPPING
