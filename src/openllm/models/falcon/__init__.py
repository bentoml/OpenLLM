from __future__ import annotations
import sys, typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule, is_torch_available

_import_structure: dict[str, list[str]] = {"configuration_falcon": ["FalconConfig", "START_FALCON_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
if t.TYPE_CHECKING:
  from .configuration_falcon import (
    DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE,
    START_FALCON_COMMAND_DOCSTRING as START_FALCON_COMMAND_DOCSTRING,
    FalconConfig as FalconConfig,
  )
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError: pass
else:
  _import_structure["modeling_falcon"] = ["Falcon"]
  if t.TYPE_CHECKING: from .modeling_falcon import Falcon as Falcon

sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure)
