from __future__ import annotations
import typing as t
from collections import OrderedDict

from openllm_core.config import CONFIG_MAPPING_NAMES

from .factory import BaseAutoLLMClass
from .factory import _LazyAutoMapping

MODEL_VLLM_MAPPING_NAMES = OrderedDict([('baichuan', 'VLLMBaichuan'), ('dolly_v2', 'VLLMDollyV2'), ('falcon', 'VLLMFalcon'), ('gpt_neox', 'VLLMGPTNeoX'), ('mpt', 'VLLMMPT'),
                                        ('opt', 'VLLMOPT'), ('stablelm', 'VLLMStableLM'), ('starcoder', 'VLLMStarCoder'), ('llama', 'VLLMLlama')])
MODEL_VLLM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_VLLM_MAPPING_NAMES)

class AutoVLLM(BaseAutoLLMClass):
  _model_mapping: t.ClassVar = MODEL_VLLM_MAPPING
