from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_cpm_kernels_available
from ...utils import is_torch_available

_import_structure: dict[str, list[str]] = {"configuration_chatglm": ["ChatGLMConfig", "START_CHATGLM_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
try:
  if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_chatglm"] = ["ChatGLM"]
if t.TYPE_CHECKING:
  from .configuration_chatglm import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_chatglm import START_CHATGLM_COMMAND_DOCSTRING as START_CHATGLM_COMMAND_DOCSTRING
  from .configuration_chatglm import ChatGLMConfig as ChatGLMConfig
  try:
    if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_chatglm import ChatGLM as ChatGLM
else:
  sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
