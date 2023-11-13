from __future__ import annotations
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import typing as t

if t.TYPE_CHECKING:
  from collections import OrderedDict

  BackendOrderedDict = OrderedDict[str, t.Tuple[t.Callable[[], bool], str]]

logger = logging.getLogger(__name__)
OPTIONAL_DEPENDENCIES = {
  'opt',
  'flan-t5',
  'vllm',
  'fine-tune',
  'ggml',
  'agents',
  'openai',
  'playground',
  'gptq',
  'grpc',
  'awq',
}
ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({'AUTO'})
USE_TORCH = os.environ.get('USE_TORCH', 'AUTO').upper()
USE_VLLM = os.environ.get('USE_VLLM', 'AUTO').upper()


def _is_package_available(package: str) -> bool:
  _package_available = importlib.util.find_spec(package) is not None
  if _package_available:
    try:
      importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      _package_available = False
  return _package_available


_torch_available = importlib.util.find_spec('torch') is not None
_vllm_available = importlib.util.find_spec('vllm') is not None
_transformers_available = _is_package_available('transformers')
_grpc_available = importlib.util.find_spec('grpc') is not None
_bentoml_available = _is_package_available('bentoml')
_peft_available = _is_package_available('peft')
_bitsandbytes_available = _is_package_available('bitsandbytes')
_jupyter_available = _is_package_available('jupyter')
_jupytext_available = _is_package_available('jupytext')
_notebook_available = _is_package_available('notebook')
_autogptq_available = _is_package_available('auto_gptq')
_autoawq_available = importlib.util.find_spec('awq') is not None


def is_bentoml_available() -> bool:
  return _bentoml_available


def is_transformers_available() -> bool:
  return _transformers_available


def is_grpc_available() -> bool:
  return _grpc_available


def is_optimum_supports_gptq() -> bool:
  from . import pkg

  return pkg.pkg_version_info('optimum')[:2] >= (0, 12)


def is_jupyter_available() -> bool:
  return _jupyter_available


def is_jupytext_available() -> bool:
  return _jupytext_available


def is_notebook_available() -> bool:
  return _notebook_available


def is_peft_available() -> bool:
  return _peft_available


def is_bitsandbytes_available() -> bool:
  return _bitsandbytes_available


def is_autogptq_available() -> bool:
  return _autogptq_available


def is_torch_available() -> bool:
  global _torch_available
  if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and _torch_available:
    try:
      importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
      _torch_available = False
  return _torch_available


def is_autoawq_available() -> bool:
  global _autoawq_available
  try:
    importlib.metadata.version('autoawq')
  except importlib.metadata.PackageNotFoundError:
    _autoawq_available = False
  return _autoawq_available


def is_vllm_available() -> bool:
  global _vllm_available
  if USE_VLLM in ENV_VARS_TRUE_AND_AUTO_VALUES and _vllm_available:
    try:
      importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
      _vllm_available = False
  return _vllm_available
