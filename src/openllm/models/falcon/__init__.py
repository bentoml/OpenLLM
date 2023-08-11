from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_torch_available

_import_structure: dict[str, list[str]] = {"configuration_falcon": ["FalconConfig", "START_FALCON_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_falcon"] = ["Falcon"]
if t.TYPE_CHECKING:
  from .configuration_falcon import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_falcon import START_FALCON_COMMAND_DOCSTRING as START_FALCON_COMMAND_DOCSTRING
  from .configuration_falcon import FalconConfig as FalconConfig
  try:
    if not is_torch_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_falcon import Falcon as Falcon
else:
  sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
