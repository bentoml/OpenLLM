from __future__ import annotations
import sys, typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_flax_available, is_tf_available, is_torch_available

_import_structure: dict[str, list[str]] = {"configuration_flan_t5": ["FlanT5Config", "START_FLAN_T5_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
if t.TYPE_CHECKING:
  from .configuration_flan_t5 import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    START_FLAN_T5_COMMAND_DOCSTRING as START_FLAN_T5_COMMAND_DOCSTRING,
    FlanT5Config as FlanT5Config,
  )
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_flan_t5"] = ["FlanT5"]
  if t.TYPE_CHECKING: from .modeling_flan_t5 import FlanT5 as FlanT5
try:
  if not is_flax_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_flax_flan_t5"] = ["FlaxFlanT5"]
  if t.TYPE_CHECKING: from .modeling_flax_flan_t5 import FlaxFlanT5 as FlaxFlanT5
try:
  if not is_tf_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_tf_flan_t5"] = ["TFFlanT5"]
  if t.TYPE_CHECKING: from .modeling_tf_flan_t5 import TFFlanT5 as TFFlanT5

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
