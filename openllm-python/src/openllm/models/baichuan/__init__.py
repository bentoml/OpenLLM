from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_cpm_kernels_available
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_baichuan import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_baichuan import START_BAICHUAN_COMMAND_DOCSTRING as START_BAICHUAN_COMMAND_DOCSTRING
from openllm_core.config.configuration_baichuan import BaichuanConfig as BaichuanConfig

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_baichuan'] = ['Baichuan']
  if t.TYPE_CHECKING: from .modeling_baichuan import Baichuan as Baichuan
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_baichuan'] = ['VLLMBaichuan']
  if t.TYPE_CHECKING: from .modeling_vllm_baichuan import VLLMBaichuan as VLLMBaichuan

sys.modules[__name__] = LazyModule(__name__, globals()['__file__'], _import_structure)
