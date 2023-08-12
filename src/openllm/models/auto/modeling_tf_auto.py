from __future__ import annotations
import typing as t
from collections import OrderedDict
from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import BaseAutoLLMClass, _LazyAutoMapping
if t.TYPE_CHECKING:
  import transformers, openllm
  from collections import OrderedDict

MODEL_TF_MAPPING_NAMES = OrderedDict([("flan_t5", "TFFlanT5"), ("opt", "TFOPT")])
MODEL_TF_MAPPING: _LazyAutoMapping[t.Type[openllm.LLMConfig], t.Type[openllm.LLM["transformers.TFPreTrainedModel", "transformers.PreTrainedTokenizer"]]] = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_TF_MAPPING_NAMES)
class AutoTFLLM(BaseAutoLLMClass):
  _model_mapping = MODEL_TF_MAPPING
