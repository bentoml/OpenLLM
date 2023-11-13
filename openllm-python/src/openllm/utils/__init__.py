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
  from openllm_core.utils import (
    DEBUG as DEBUG,
    DEBUG_ENV_VAR as DEBUG_ENV_VAR,
    DEV_DEBUG_VAR as DEV_DEBUG_VAR,
    ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES,
    MYPY as MYPY,
    OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES,
    QUIET_ENV_VAR as QUIET_ENV_VAR,
    SHOW_CODEGEN as SHOW_CODEGEN,
    LazyLoader as LazyLoader,
    LazyModule as LazyModule,
    ReprMixin as ReprMixin,
    VersionInfo as VersionInfo,
    analytics as analytics,
    calc_dir_size as calc_dir_size,
    check_bool_env as check_bool_env,
    codegen as codegen,
    configure_logging as configure_logging,
    dantic as dantic,
    field_env_key as field_env_key,
    first_not_none as first_not_none,
    flatten_attrs as flatten_attrs,
    gen_random_uuid as gen_random_uuid,
    generate_context as generate_context,
    generate_hash_from_file as generate_hash_from_file,
    get_debug_mode as get_debug_mode,
    get_disable_warnings as get_disable_warnings,
    get_quiet_mode as get_quiet_mode,
    in_notebook as in_notebook,
    is_autoawq_available as is_autoawq_available,
    is_autogptq_available as is_autogptq_available,
    is_bentoml_available as is_bentoml_available,
    is_bitsandbytes_available as is_bitsandbytes_available,
    is_grpc_available as is_grpc_available,
    is_jupyter_available as is_jupyter_available,
    is_jupytext_available as is_jupytext_available,
    is_notebook_available as is_notebook_available,
    is_optimum_supports_gptq as is_optimum_supports_gptq,
    is_peft_available as is_peft_available,
    is_torch_available as is_torch_available,
    is_transformers_available as is_transformers_available,
    is_vllm_available as is_vllm_available,
    lenient_issubclass as lenient_issubclass,
    reserve_free_port as reserve_free_port,
    resolve_filepath as resolve_filepath,
    resolve_user_filepath as resolve_user_filepath,
    serde as serde,
    set_debug_mode as set_debug_mode,
    set_disable_warnings as set_disable_warnings,
    set_quiet_mode as set_quiet_mode,
    validate_is_path as validate_is_path,
  )
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
