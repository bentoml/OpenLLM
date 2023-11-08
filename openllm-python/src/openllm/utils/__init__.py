"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations
import functools
import typing as t

import openllm_core

if t.TYPE_CHECKING:
  import openllm

  from openllm_core.utils import OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES
  from openllm_core.utils import EnvVarMixin as EnvVarMixin
  from openllm_core.utils import ReprMixin as ReprMixin
  from openllm_core.utils import analytics as analytics
  from openllm_core.utils import codegen as codegen
  from openllm_core.utils import dantic as dantic
  from openllm_core.utils import is_autoawq_available as is_autoawq_available
  from openllm_core.utils import is_autogptq_available as is_autogptq_available
  from openllm_core.utils import is_bentoml_available as is_bentoml_available
  from openllm_core.utils import is_bitsandbytes_available as is_bitsandbytes_available
  from openllm_core.utils import is_grpc_available as is_grpc_available
  from openllm_core.utils import is_jupyter_available as is_jupyter_available
  from openllm_core.utils import is_jupytext_available as is_jupytext_available
  from openllm_core.utils import is_notebook_available as is_notebook_available
  from openllm_core.utils import is_optimum_supports_gptq as is_optimum_supports_gptq
  from openllm_core.utils import is_peft_available as is_peft_available
  from openllm_core.utils import is_torch_available as is_torch_available
  from openllm_core.utils import is_transformers_available as is_transformers_available
  from openllm_core.utils import is_vllm_available as is_vllm_available
  from openllm_core.utils import serde as serde
  from openllm_core.utils.serde import converter as converter

def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> dict[str, t.Any]:
  return {'backend': llm.__llm_backend__, 'framework': 'openllm', 'model_name': llm.config['model_name'], 'architecture': llm.config['architecture'], 'serialisation': llm._serialisation}

def available_devices() -> tuple[str, ...]:
  """Return available GPU under system. Currently only supports NVIDIA GPUs."""
  from .._strategies import NvidiaGpuResource
  return tuple(NvidiaGpuResource.from_system())

@functools.lru_cache(maxsize=1)
def device_count() -> int:
  return len(available_devices())

__all__ = ['generate_labels', 'available_devices', 'device_count']

def __dir__() -> t.Sequence[str]:
  return sorted(__all__)

def __getattr__(it: str) -> t.Any:
  if hasattr(openllm_core.utils, it): return getattr(openllm_core.utils, it)
  else: raise AttributeError(f'module {__name__} has no attribute {it}')
