from __future__ import annotations

import importlib, importlib.metadata, importlib.util, os, inspect, typing as t
from .codegen import _make_method
from ._constants import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES

ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({'AUTO'})
USE_VLLM = os.getenv('USE_VLLM', 'AUTO').upper()


def _has_package(package: str) -> bool:
  _package_available = importlib.util.find_spec(package) is not None
  if _package_available:
    try:
      importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      _package_available = False
  return _package_available


_grpc_available = _has_package('grpc')
_triton_available = _has_package('triton')
_torch_available = _has_package('torch')
_bentoml_available = _has_package('bentoml')
_transformers_available = _has_package('transformers')
_peft_available = _has_package('peft')
_bitsandbytes_available = _has_package('bitsandbytes')
_flash_attn_2_available = _has_package('flash_attn')
_jupyter_available = _has_package('jupyter')
_jupytext_available = _has_package('jupytext')
_notebook_available = _has_package('notebook')
_autogptq_available = _has_package('auto_gptq')

_availables = {
  k[1:]: v for k, v in locals().items() if k.startswith('_') and not inspect.isfunction(v) and k.endswith('_available')
}
caller = {
  f'is_{k}': _make_method(
    f'is_{k}', f'def is_{k}() -> bool:\n  global _{k}\n  return _{k}\n', f'generated_file_{k}', {f'_{k}': v}
  )
  for k, v in _availables.items()
}

_autoawq_available = importlib.util.find_spec('awq') is not None


def is_autoawq_available() -> bool:
  global _autoawq_available
  try:
    importlib.metadata.version('autoawq')
  except importlib.metadata.PackageNotFoundError:
    _autoawq_available = False
  return _autoawq_available


_vllm_available = importlib.util.find_spec('vllm') is not None


def is_vllm_available() -> bool:
  global _vllm_available
  if USE_VLLM in ENV_VARS_TRUE_AND_AUTO_VALUES or _vllm_available:
    try:
      importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
      _vllm_available = False
  return _vllm_available


def __dir__() -> list[str]:
  return [*list(caller.keys()), 'is_autoawq_available', 'is_vllm_available', 'USE_VLLM', 'ENV_VARS_TRUE_VALUES']


def __getattr__(it: t.Any) -> t.Any:
  if it in caller:
    return caller[it]
  raise AttributeError(f'module {__name__!r} has no attribute {it!r}')
