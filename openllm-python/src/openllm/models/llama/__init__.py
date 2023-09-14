from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_llama import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_llama import PROMPT_MAPPING as PROMPT_MAPPING
from openllm_core.config.configuration_llama import START_LLAMA_COMMAND_DOCSTRING as START_LLAMA_COMMAND_DOCSTRING
from openllm_core.config.configuration_llama import LlamaConfig as LlamaConfig

_import_structure: dict[str, list[str]] = {}
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_llama'] = ['VLLMLlama']
  if t.TYPE_CHECKING: from .modeling_vllm_llama import VLLMLlama as VLLMLlama
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_llama'] = ['Llama']
  if t.TYPE_CHECKING: from .modeling_llama import Llama as Llama

sys.modules[__name__] = LazyModule(__name__,
                                   globals()['__file__'],
                                   _import_structure,
                                   extra_objects={
                                       'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE,
                                       'START_LLAMA_COMMAND_DOCSTRING': START_LLAMA_COMMAND_DOCSTRING,
                                       'LlamaConfig': LlamaConfig,
                                       'PROMPT_MAPPING': PROMPT_MAPPING
                                   })
