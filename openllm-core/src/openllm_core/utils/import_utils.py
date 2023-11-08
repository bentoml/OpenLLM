"""Some imports utils are vendorred from transformers/utils/import_utils.py for performance reasons."""
from __future__ import annotations
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import typing as t

import inflection

import openllm_core

from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralString
from openllm_core._typing_compat import overload

from .lazy import LazyLoader
from .representation import ReprMixin

if t.TYPE_CHECKING:
  from collections import OrderedDict
  BackendOrderedDict = OrderedDict[str, t.Tuple[t.Callable[[], bool], str]]

logger = logging.getLogger(__name__)
OPTIONAL_DEPENDENCIES = {'opt', 'flan-t5', 'vllm', 'fine-tune', 'ggml', 'agents', 'openai', 'playground', 'gptq', 'grpc', 'awq'}
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
_grpc_health_available = importlib.util.find_spec('grpc_health') is not None
_bentoml_available = _is_package_available('bentoml')
_peft_available = _is_package_available('peft')
_einops_available = _is_package_available('einops')
_cpm_kernel_available = _is_package_available('cpm_kernels')
_bitsandbytes_available = _is_package_available('bitsandbytes')
_datasets_available = _is_package_available('datasets')
_triton_available = _is_package_available('triton')
_jupyter_available = _is_package_available('jupyter')
_jupytext_available = _is_package_available('jupytext')
_notebook_available = _is_package_available('notebook')
_autogptq_available = _is_package_available('auto_gptq')
_autoawq_available = _is_package_available('awq')
_sentencepiece_available = _is_package_available('sentencepiece')
_xformers_available = _is_package_available('xformers')
_fairscale_available = _is_package_available('fairscale')

def is_bentoml_available() -> bool:
  return _bentoml_available

def is_transformers_available() -> bool:
  return _transformers_available

def is_grpc_available() -> bool:
  return _grpc_available

def is_grpc_health_available() -> bool:
  return _grpc_health_available

def is_optimum_supports_gptq() -> bool:
  from . import pkg
  return pkg.pkg_version_info('optimum')[:2] >= (0, 12)

def is_jupyter_available() -> bool:
  return _jupyter_available

def is_jupytext_available() -> bool:
  return _jupytext_available

def is_notebook_available() -> bool:
  return _notebook_available

def is_triton_available() -> bool:
  return _triton_available

def is_datasets_available() -> bool:
  return _datasets_available

def is_peft_available() -> bool:
  return _peft_available

def is_einops_available() -> bool:
  return _einops_available

def is_cpm_kernels_available() -> bool:
  return _cpm_kernel_available

def is_bitsandbytes_available() -> bool:
  return _bitsandbytes_available

def is_autogptq_available() -> bool:
  return _autogptq_available

def is_sentencepiece_available() -> bool:
  return _sentencepiece_available

def is_xformers_available() -> bool:
  return _xformers_available

def is_fairscale_available() -> bool:
  return _fairscale_available

def is_torch_available() -> bool:
  global _torch_available
  if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and _torch_available:
    try:
      importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
      _torch_available = False
  return _torch_available

def is_autoawq_available() -> bool:
  return _autoawq_available

def is_vllm_available() -> bool:
  global _vllm_available
  if USE_VLLM in ENV_VARS_TRUE_AND_AUTO_VALUES and _vllm_available:
    try:
      importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
      _vllm_available = False
  return _vllm_available

class EnvVarMixin(ReprMixin):
  model_name: str
  config: str
  model_id: str
  quantize: str
  backend: str

  @overload
  def __getitem__(self, item: t.Literal['config']) -> str:
    ...

  @overload
  def __getitem__(self, item: t.Literal['model_id']) -> str:
    ...

  @overload
  def __getitem__(self, item: t.Literal['quantize']) -> str:
    ...

  @overload
  def __getitem__(self, item: t.Literal['backend']) -> str:
    ...

  @overload
  def __getitem__(self, item: t.Literal['backend_value']) -> LiteralBackend:
    ...

  @overload
  def __getitem__(self, item: t.Literal['quantize_value']) -> t.Literal['int8', 'int4', 'gptq'] | None:
    ...

  @overload
  def __getitem__(self, item: t.Literal['model_id_value']) -> str | None:
    ...

  def __getitem__(self, item: str | t.Any) -> t.Any:
    if item.endswith('_value') and hasattr(self, f'_{item}'): return object.__getattribute__(self, f'_{item}')()
    elif hasattr(self, item): return getattr(self, item)
    raise KeyError(f'Key {item} not found in {self}')

  def __init__(self, model_name: str, backend: LiteralBackend = 'pt', model_id: str | None = None, quantize: LiteralString | None = None) -> None:
    """EnvVarMixin is a mixin class that returns the value extracted from environment variables."""
    from openllm_core.utils import field_env_key
    self.model_name = inflection.underscore(model_name)
    self._backend = backend
    self._model_id = model_id
    self._quantize = quantize
    for att in {'config', 'model_id', 'quantize', 'backend'}:
      setattr(self, att, field_env_key(att.upper()))

  def _quantize_value(self) -> t.Literal['int8', 'int4', 'gptq'] | None:
    from . import first_not_none
    return t.cast(t.Optional[t.Literal['int8', 'int4', 'gptq']], first_not_none(os.environ.get(self['quantize']), default=self._quantize))

  def _backend_value(self) -> LiteralBackend:
    from . import first_not_none
    return t.cast(LiteralBackend, first_not_none(os.environ.get(self['backend']), default=self._backend))

  def _model_id_value(self) -> str | None:
    from . import first_not_none
    return first_not_none(os.environ.get(self['model_id']), default=self._model_id)

  @property
  def __repr_keys__(self) -> set[str]:
    return {'config', 'model_id', 'quantize', 'backend'}

  @property
  def start_docstring(self) -> str:
    return getattr(openllm_core.config, f'START_{self.model_name.upper()}_COMMAND_DOCSTRING')

  @property
  def module(self) -> LazyLoader:
    return LazyLoader(f'configuration_{self.model_name}', globals(), f'openllm_core.config.configuration_{self.model_name}')
