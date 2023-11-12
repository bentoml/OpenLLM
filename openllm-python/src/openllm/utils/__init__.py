"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""

from __future__ import annotations
import functools
import importlib.metadata
import typing as t

import openllm_core


if t.TYPE_CHECKING:
  import openllm

  from openllm_core.utils import DEBUG as DEBUG
  from openllm_core.utils import DEBUG_ENV_VAR as DEBUG_ENV_VAR
  from openllm_core.utils import DEV_DEBUG_VAR as DEV_DEBUG_VAR
  from openllm_core.utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
  from openllm_core.utils import MYPY as MYPY
  from openllm_core.utils import OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES
  from openllm_core.utils import QUIET_ENV_VAR as QUIET_ENV_VAR
  from openllm_core.utils import SHOW_CODEGEN as SHOW_CODEGEN
  from openllm_core.utils import LazyLoader as LazyLoader
  from openllm_core.utils import LazyModule as LazyModule
  from openllm_core.utils import ReprMixin as ReprMixin
  from openllm_core.utils import VersionInfo as VersionInfo
  from openllm_core.utils import analytics as analytics
  from openllm_core.utils import calc_dir_size as calc_dir_size
  from openllm_core.utils import check_bool_env as check_bool_env
  from openllm_core.utils import codegen as codegen
  from openllm_core.utils import configure_logging as configure_logging
  from openllm_core.utils import dantic as dantic
  from openllm_core.utils import field_env_key as field_env_key
  from openllm_core.utils import first_not_none as first_not_none
  from openllm_core.utils import flatten_attrs as flatten_attrs
  from openllm_core.utils import gen_random_uuid as gen_random_uuid
  from openllm_core.utils import generate_context as generate_context
  from openllm_core.utils import generate_hash_from_file as generate_hash_from_file
  from openllm_core.utils import get_debug_mode as get_debug_mode
  from openllm_core.utils import get_disable_warnings as get_disable_warnings
  from openllm_core.utils import get_quiet_mode as get_quiet_mode
  from openllm_core.utils import in_notebook as in_notebook
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
  from openllm_core.utils import lenient_issubclass as lenient_issubclass
  from openllm_core.utils import reserve_free_port as reserve_free_port
  from openllm_core.utils import resolve_filepath as resolve_filepath
  from openllm_core.utils import resolve_user_filepath as resolve_user_filepath
  from openllm_core.utils import serde as serde
  from openllm_core.utils import set_debug_mode as set_debug_mode
  from openllm_core.utils import set_disable_warnings as set_disable_warnings
  from openllm_core.utils import set_quiet_mode as set_quiet_mode
  from openllm_core.utils import validate_is_path as validate_is_path
  from openllm_core.utils.serde import converter as converter


def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> dict[str, t.Any]:
  return {
    'backend': llm.__llm_backend__,
    'framework': 'openllm',
    'model_name': llm.config['model_name'],
    'architecture': llm.config['architecture'],
    'serialisation': llm._serialisation,
    **{package: importlib.metadata.version(package) for package in {'openllm', 'openllm-core', 'openllm-client'}},
  }


def available_devices() -> tuple[str, ...]:
  """Return available GPU under system. Currently only supports NVIDIA GPUs."""
  from .._strategies import NvidiaGpuResource

  return tuple(NvidiaGpuResource.from_system())


@functools.lru_cache(maxsize=1)
def device_count() -> int:
  return len(available_devices())


__all__ = ['generate_labels', 'available_devices', 'device_count']


def __dir__() -> t.Sequence[str]:
  return sorted(__all__) + sorted(dir(openllm_core.utils))


def __getattr__(it: str) -> t.Any:
  if hasattr(openllm_core.utils, it):
    return getattr(openllm_core.utils, it)
  else:
    raise AttributeError(f'module {__name__} has no attribute {it}')
