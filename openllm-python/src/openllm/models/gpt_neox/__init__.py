from __future__ import annotations
import sys
import typing as t

from openllm.exceptions import MissingDependencyError
from openllm.utils import LazyModule
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available
from openllm_core.config.configuration_gpt_neox import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
from openllm_core.config.configuration_gpt_neox import START_GPT_NEOX_COMMAND_DOCSTRING as START_GPT_NEOX_COMMAND_DOCSTRING
from openllm_core.config.configuration_gpt_neox import GPTNeoXConfig as GPTNeoXConfig

_import_structure: dict[str, list[str]] = {}
try:
  if not is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_gpt_neox'] = ['GPTNeoX']
  if t.TYPE_CHECKING: from .modeling_gpt_neox import GPTNeoX as GPTNeoX
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure['modeling_vllm_gpt_neox'] = ['VLLMGPTNeoX']
  if t.TYPE_CHECKING: from .modeling_vllm_gpt_neox import VLLMGPTNeoX as VLLMGPTNeoX

sys.modules[__name__] = LazyModule(__name__,
                                   globals()['__file__'],
                                   _import_structure,
                                   extra_objects={
                                       'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE, 'START_GPT_NEOX_COMMAND_DOCSTRING': START_GPT_NEOX_COMMAND_DOCSTRING, 'GPTNeoXConfig': GPTNeoXConfig
                                   })
