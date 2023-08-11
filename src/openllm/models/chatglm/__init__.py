from __future__ import annotations
import sys, typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_cpm_kernels_available, is_torch_available

_import_structure: dict[str, list[str]] = {"configuration_chatglm": ["ChatGLMConfig", "START_CHATGLM_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
if t.TYPE_CHECKING:
  from .configuration_chatglm import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    START_CHATGLM_COMMAND_DOCSTRING as START_CHATGLM_COMMAND_DOCSTRING,
    ChatGLMConfig as ChatGLMConfig,
  )
try:
  if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_chatglm"] = ["ChatGLM"]
  if t.TYPE_CHECKING: from .modeling_chatglm import ChatGLM as ChatGLM

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
