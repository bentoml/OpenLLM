from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_torch_available
from ...utils import is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_mpt": ["MPTConfig", "START_MPT_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE", "PROMPT_MAPPING"]}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError: pass
else: _import_structure["modeling_mpt"] = ["MPT"]
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError: pass
else: _import_structure["modeling_vllm_mpt"] = ["VLLMMPT"]
if t.TYPE_CHECKING:
  from .configuration_mpt import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_mpt import PROMPT_MAPPING as PROMPT_MAPPING
  from .configuration_mpt import START_MPT_COMMAND_DOCSTRING as START_MPT_COMMAND_DOCSTRING
  from .configuration_mpt import MPTConfig as MPTConfig
  try:
    if not is_torch_available(): raise MissingDependencyError
  except MissingDependencyError: pass
  else: from .modeling_mpt import MPT as MPT
  try:
    if not is_vllm_available(): raise MissingDependencyError
  except MissingDependencyError: pass
  else: from .modeling_vllm_mpt import VLLMMPT as VLLMMPT
else: sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
