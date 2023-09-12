from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_falcon import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_falcon import START_FALCON_COMMAND_DOCSTRING as START_FALCON_COMMAND_DOCSTRING
from openllm_core.config.configuration_falcon import FalconConfig as FalconConfig

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_falcon'] = ['Falcon']
  if t.TYPE_CHECKING: from .modeling_falcon import Falcon as Falcon
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_falcon'] = ['VLLMFalcon']
  if t.TYPE_CHECKING: from .modeling_vllm_falcon import VLLMFalcon as VLLMFalcon

sys.modules[__name__] = LazyModule(__name__, globals()['__file__'], _import_structure)
