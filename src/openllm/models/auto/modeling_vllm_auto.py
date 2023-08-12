from __future__ import annotations
import typing as t
from collections import OrderedDict
from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import BaseAutoLLMClass, _LazyAutoMapping
if t.TYPE_CHECKING:
  import vllm, transformers, openllm
  from collections import OrderedDict

MODEL_VLLM_MAPPING_NAMES = OrderedDict([("baichuan", "VLLMBaichuan"), ("dolly_v2", "VLLMDollyV2"), ("gpt_neox", "VLLMGPTNeoX"), ("mpt", "VLLMMPT"), ("opt", "VLLMOPT"), ("stablelm", "VLLMStableLM"), ("starcoder", "VLLMStarCoder"), ("llama", "VLLMLlama")])
MODEL_VLLM_MAPPING: _LazyAutoMapping[t.Type[openllm.LLMConfig], t.Type[openllm.LLM["vllm.LLMEngine", "transformers.PreTrainedTokenizer"]]] = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_VLLM_MAPPING_NAMES)
class AutoVLLM(BaseAutoLLMClass):
  _model_mapping = MODEL_VLLM_MAPPING
