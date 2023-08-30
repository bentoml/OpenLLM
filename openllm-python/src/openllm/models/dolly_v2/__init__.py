from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_dolly_v2 import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_dolly_v2 import START_DOLLY_V2_COMMAND_DOCSTRING as START_DOLLY_V2_COMMAND_DOCSTRING
from openllm_core.config.configuration_dolly_v2 import DollyV2Config as DollyV2Config

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_dolly_v2'] = ['DollyV2']
  if t.TYPE_CHECKING: from .modeling_dolly_v2 import DollyV2 as DollyV2
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_dolly_v2'] = ['VLLMDollyV2']
  if t.TYPE_CHECKING: from .modeling_vllm_dolly_v2 import VLLMDollyV2 as VLLMDollyV2

sys.modules[__name__] = LazyModule(__name__, globals()['__file__'], _import_structure)
