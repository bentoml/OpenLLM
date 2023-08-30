from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_flax_available
from openllm.utils import is_tf_available
from openllm.utils import is_torch_available
from openllm_core.config.configuration_flan_t5 import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_flan_t5 import START_FLAN_T5_COMMAND_DOCSTRING as START_FLAN_T5_COMMAND_DOCSTRING
from openllm_core.config.configuration_flan_t5 import FlanT5Config as FlanT5Config

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_flan_t5'] = ['FlanT5']
  if t.TYPE_CHECKING: from .modeling_flan_t5 import FlanT5 as FlanT5
try:
  if not is_flax_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_flax_flan_t5'] = ['FlaxFlanT5']
  if t.TYPE_CHECKING: from .modeling_flax_flan_t5 import FlaxFlanT5 as FlaxFlanT5
try:
  if not is_tf_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_tf_flan_t5'] = ['TFFlanT5']
  if t.TYPE_CHECKING: from .modeling_tf_flan_t5 import TFFlanT5 as TFFlanT5

sys.modules[__name__] = LazyModule(__name__, globals()['__file__'], _import_structure)
