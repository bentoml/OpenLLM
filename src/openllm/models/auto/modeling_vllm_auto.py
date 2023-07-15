from __future__ import annotations
import typing as t
from collections import OrderedDict

from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import BaseAutoLLMClass
from .factory import _LazyAutoMapping

if t.TYPE_CHECKING:
    import openllm

MODEL_VLLM_MAPPING_NAMES = OrderedDict()

MODEL_VLLM_MAPPING: dict[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]] = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_VLLM_MAPPING_NAMES
)


class AutoVLLM(BaseAutoLLMClass):
    _model_mapping = MODEL_VLLM_MAPPING
