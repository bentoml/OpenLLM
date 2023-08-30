from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_starcoder import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_starcoder import START_STARCODER_COMMAND_DOCSTRING as START_STARCODER_COMMAND_DOCSTRING
from openllm_core.config.configuration_starcoder import StarCoderConfig as StarCoderConfig

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_starcoder'] = ['StarCoder']
  if t.TYPE_CHECKING: from .modeling_starcoder import StarCoder as StarCoder
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_starcoder'] = ['VLLMStarCoder']
  if t.TYPE_CHECKING: from .modeling_vllm_starcoder import VLLMStarCoder as VLLMStarCoder

sys.modules[__name__] = LazyModule(__name__, globals()['__file__'], _import_structure)
