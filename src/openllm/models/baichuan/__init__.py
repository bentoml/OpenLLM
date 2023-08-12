from __future__ import annotations
import sys, typing as t
from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_cpm_kernels_available, is_torch_available, is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_baichuan": ["BaichuanConfig", "START_BAICHUAN_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
if t.TYPE_CHECKING:
  from .configuration_baichuan import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    START_BAICHUAN_COMMAND_DOCSTRING as START_BAICHUAN_COMMAND_DOCSTRING,
    BaichuanConfig as BaichuanConfig,
  )
try:
  if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_baichuan"] = ["Baichuan"]
  if t.TYPE_CHECKING: from .modeling_baichuan import Baichuan as Baichuan
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_vllm_baichuan"] = ["VLLMBaichuan"]
  if t.TYPE_CHECKING: from .modeling_vllm_baichuan import VLLMBaichuan as VLLMBaichuan

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
