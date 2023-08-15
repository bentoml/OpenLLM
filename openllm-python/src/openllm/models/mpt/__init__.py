from __future__ import annotations
import sys, typing as t
from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_torch_available, is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_mpt": ["MPTConfig", "START_MPT_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE", "PROMPT_MAPPING"]}
if t.TYPE_CHECKING:
  from .configuration_mpt import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    PROMPT_MAPPING as PROMPT_MAPPING,
    START_MPT_COMMAND_DOCSTRING as START_MPT_COMMAND_DOCSTRING,
    MPTConfig as MPTConfig,
  )
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_mpt"] = ["MPT"]
  if t.TYPE_CHECKING: from .modeling_mpt import MPT as MPT
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_vllm_mpt"] = ["VLLMMPT"]
  if t.TYPE_CHECKING: from .modeling_vllm_mpt import VLLMMPT as VLLMMPT

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
