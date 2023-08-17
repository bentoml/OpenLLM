from __future__ import annotations
import typing as t
from collections import OrderedDict
from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import BaseAutoLLMClass, _LazyAutoMapping

MODEL_GGML_MAPPING_NAMES = OrderedDict([("llama", "GGMLLlama")])
MODEL_GGML_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_GGML_MAPPING_NAMES)
class AutoGGML(BaseAutoLLMClass):
  _model_mapping: t.ClassVar = MODEL_GGML_MAPPING
