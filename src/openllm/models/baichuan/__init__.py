from __future__ import annotations
import sys
import typing as t
from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_cpm_kernels_available
from ...utils import is_torch_available
from ...utils import is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_baichuan": ["BaichuanConfig", "START_BAICHUAN_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"]}
try:
  if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_baichuan"] = ["Baichuan"]
try:
  if not is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  pass
else:
  _import_structure["modeling_vllm_baichuan"] = ["VLLMBaichuan"]
if t.TYPE_CHECKING:
  from .configuration_baichuan import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
  from .configuration_baichuan import START_BAICHUAN_COMMAND_DOCSTRING as START_BAICHUAN_COMMAND_DOCSTRING
  from .configuration_baichuan import BaichuanConfig as BaichuanConfig

  try:
    if not is_torch_available() or not is_cpm_kernels_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_baichuan import Baichuan as Baichuan
  try:
    if not is_vllm_available(): raise MissingDependencyError
  except MissingDependencyError:
    pass
  else:
    from .modeling_vllm_baichuan import VLLMBaichuan as VLLMBaichuan
else:
  sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
