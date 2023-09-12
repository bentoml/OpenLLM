from __future__ import annotations
import typing as t
from collections import OrderedDict

from openllm_core.config import CONFIG_MAPPING_NAMES

from .factory import BaseAutoLLMClass
from .factory import _LazyAutoMapping

MODEL_FLAX_MAPPING_NAMES = OrderedDict([('flan_t5', 'FlaxFlanT5'), ('opt', 'FlaxOPT')])
MODEL_FLAX_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FLAX_MAPPING_NAMES)

class AutoFlaxLLM(BaseAutoLLMClass):
  _model_mapping: t.ClassVar = MODEL_FLAX_MAPPING
