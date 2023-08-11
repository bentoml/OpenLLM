from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_torch_available
from ...utils import is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_llama": ["LlamaConfig", "START_LLAMA_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE", "PROMPT_MAPPING"]}
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_vllm_llama"] = ["VLLMLlama"]
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_llama"] = ["Llama"]
if t.TYPE_CHECKING:
  from .configuration_llama import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_llama import PROMPT_MAPPING as PROMPT_MAPPING
  from .configuration_llama import START_LLAMA_COMMAND_DOCSTRING as START_LLAMA_COMMAND_DOCSTRING
  from .configuration_llama import LlamaConfig as LlamaConfig
  try:
    if not is_vllm_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_vllm_llama import VLLMLlama as VLLMLlama
  try:
    if not is_torch_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_llama import Llama as Llama
else:
  sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
