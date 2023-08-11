from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_torch_available, is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_gpt_neox": ["GPTNeoXConfig", "START_GPT_NEOX_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_gpt_neox"] = ["GPTNeoX"]
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_vllm_gpt_neox"] = ["VLLMGPTNeoX"]
if t.TYPE_CHECKING:
  from .configuration_gpt_neox import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_gpt_neox import START_GPT_NEOX_COMMAND_DOCSTRING as START_GPT_NEOX_COMMAND_DOCSTRING
  from .configuration_gpt_neox import GPTNeoXConfig as GPTNeoXConfig
  try:
    if not is_torch_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_gpt_neox import GPTNeoX as GPTNeoX
  try:
    if not is_vllm_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_vllm_gpt_neox import VLLMGPTNeoX as VLLMGPTNeoX
else:
  sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
