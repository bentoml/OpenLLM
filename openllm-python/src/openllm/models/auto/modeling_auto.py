from __future__ import annotations
import typing as t
from collections import OrderedDict

from openllm_core.config import CONFIG_MAPPING_NAMES

from .factory import BaseAutoLLMClass
from .factory import _LazyAutoMapping

MODEL_MAPPING_NAMES = OrderedDict([('chatglm', 'ChatGLM'), ('dolly_v2', 'DollyV2'), ('falcon', 'Falcon'), ('flan_t5', 'FlanT5'), ('gpt_neox', 'GPTNeoX'), ('llama', 'Llama'), ('mpt', 'MPT'),
                                   ('opt', 'OPT'), ('stablelm', 'StableLM'), ('starcoder', 'StarCoder'), ('baichuan', 'Baichuan')])
MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)

class AutoLLM(BaseAutoLLMClass):
  _model_mapping: t.ClassVar = MODEL_MAPPING
