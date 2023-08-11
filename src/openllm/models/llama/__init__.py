from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_torch_available, is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_llama": ["LlamaConfig", "START_LLAMA_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE", "PROMPT_MAPPING"]}
if t.TYPE_CHECKING:
  from .configuration_llama import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    PROMPT_MAPPING as PROMPT_MAPPING,
    START_LLAMA_COMMAND_DOCSTRING as START_LLAMA_COMMAND_DOCSTRING,
    LlamaConfig as LlamaConfig,
  )
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_vllm_llama"] = ["VLLMLlama"]
  if t.TYPE_CHECKING: from .modeling_vllm_llama import VLLMLlama as VLLMLlama
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_llama"] = ["Llama"]
  if t.TYPE_CHECKING: from .modeling_llama import Llama as Llama

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
