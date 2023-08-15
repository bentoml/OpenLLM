from __future__ import annotations
import sys, typing as t
from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_flax_available, is_tf_available, is_torch_available, is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_opt": ["OPTConfig", "START_OPT_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
if t.TYPE_CHECKING:
  from .configuration_opt import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    START_OPT_COMMAND_DOCSTRING as START_OPT_COMMAND_DOCSTRING,
    OPTConfig as OPTConfig,
  )
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_opt"] = ["OPT"]
  if t.TYPE_CHECKING: from .modeling_opt import OPT as OPT
try:
  if not is_flax_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_flax_opt"] = ["FlaxOPT"]
  if t.TYPE_CHECKING: from .modeling_flax_opt import FlaxOPT as FlaxOPT
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_vllm_opt"] = ["VLLMOPT"]
  if t.TYPE_CHECKING: from .modeling_vllm_opt import VLLMOPT as VLLMOPT
try:
  if not is_tf_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_tf_opt"] = ["TFOPT"]
  if t.TYPE_CHECKING: from .modeling_tf_opt import TFOPT as TFOPT

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
