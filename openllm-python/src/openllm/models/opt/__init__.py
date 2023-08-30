from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_flax_available
from openllm.utils import is_tf_available
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_opt import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_opt import START_OPT_COMMAND_DOCSTRING as START_OPT_COMMAND_DOCSTRING
from openllm_core.config.configuration_opt import OPTConfig as OPTConfig

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_opt'] = ['OPT']
  if t.TYPE_CHECKING: from .modeling_opt import OPT as OPT
try:
  if not is_flax_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_flax_opt'] = ['FlaxOPT']
  if t.TYPE_CHECKING: from .modeling_flax_opt import FlaxOPT as FlaxOPT
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_opt'] = ['VLLMOPT']
  if t.TYPE_CHECKING: from .modeling_vllm_opt import VLLMOPT as VLLMOPT
try:
  if not is_tf_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_tf_opt'] = ['TFOPT']
  if t.TYPE_CHECKING: from .modeling_tf_opt import TFOPT as TFOPT

sys.modules[__name__] = LazyModule(__name__, globals()['__file__'], _import_structure)
