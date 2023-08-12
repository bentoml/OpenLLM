from __future__ import annotations
import typing as t
from collections import OrderedDict
from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import BaseAutoLLMClass, _LazyAutoMapping
if t.TYPE_CHECKING:
  import transformers, openllm
  from collections import OrderedDict

MODEL_FLAX_MAPPING_NAMES = OrderedDict([("flan_t5", "FlaxFlanT5"), ("opt", "FlaxOPT")])
MODEL_FLAX_MAPPING: OrderedDict[t.Type[openllm.LLMConfig], t.Type[openllm.LLM["transformers.FlaxPreTrainedModel", "transformers.PreTrainedTokenizer"]]] = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FLAX_MAPPING_NAMES)
class AutoFlaxLLM(BaseAutoLLMClass):
  _model_mapping = MODEL_FLAX_MAPPING
