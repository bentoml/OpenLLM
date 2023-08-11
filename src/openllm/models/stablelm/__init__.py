from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_torch_available, is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_stablelm": ["StableLMConfig", "START_STABLELM_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_stablelm"] = ["StableLM"]
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_vllm_stablelm"] = ["VLLMStableLM"]
if t.TYPE_CHECKING:
  from .configuration_stablelm import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_stablelm import START_STABLELM_COMMAND_DOCSTRING as START_STABLELM_COMMAND_DOCSTRING
  from .configuration_stablelm import StableLMConfig as StableLMConfig
  try:
    if not is_torch_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_stablelm import StableLM as StableLM
  try:
    if not is_vllm_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_vllm_stablelm import VLLMStableLM as VLLMStableLM
else:
  sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
