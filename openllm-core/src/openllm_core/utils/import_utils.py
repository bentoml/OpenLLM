'''Some imports utils are vendorred from transformers/utils/import_utils.py for performance reasons.'''
from __future__ import annotations
import abc
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import typing as t

from collections import OrderedDict

import inflection

import openllm_core

from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import pkg
from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralString
from openllm_core._typing_compat import overload

from .representation import ReprMixin

if t.TYPE_CHECKING:
  BackendOrderedDict = OrderedDict[str, t.Tuple[t.Callable[[], bool], str]]

logger = logging.getLogger(__name__)
OPTIONAL_DEPENDENCIES = {'opt', 'flan-t5', 'vllm', 'fine-tune', 'ggml', 'agents', 'openai', 'playground', 'gptq', 'grpc'}
ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({'AUTO'})
USE_TORCH = os.environ.get('USE_TORCH', 'AUTO').upper()
FORCE_TF_AVAILABLE = os.environ.get('FORCE_TF_AVAILABLE', 'AUTO').upper()

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
_sentencepiece_available = _is_package_available('sentencepiece')
_xformers_available = _is_package_available('xformers')
_fairscale_available = _is_package_available('fairscale')

def is_transformers_available() -> bool:
  return _transformers_available

def is_grpc_available() -> bool:
  return _grpc_available

def is_grpc_health_available() -> bool:
  return _grpc_health_available

def is_optimum_supports_gptq() -> bool:
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

def is_vllm_available() -> bool:
  return _vllm_available

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

VLLM_IMPORT_ERROR_WITH_PYTORCH = '''\
{0} requires the vLLM library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "VLLM", but are otherwise identically named to our PyTorch classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use vLLM, please follow the instructions on the
installation page https://github.com/vllm-project/vllm that match your environment.
'''
VLLM_IMPORT_ERROR_WITH_TF = '''\
{0} requires the vLLM library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to the PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use vLLM, please follow the instructions on the
installation page https://github.com/vllm-project/vllm that match your environment.
'''
PYTORCH_IMPORT_ERROR = '''{0} requires the PyTorch library but it was not found in your environment.
Checkout the instructions on the installation page: https://pytorch.org/get-started/locally/ and follow the
ones that match your environment. Please note that you may need to restart your runtime after installation.
'''
VLLM_IMPORT_ERROR = '''{0} requires the vLLM library but it was not found in your environment.
Checkout the instructions on the installation page: https://github.com/vllm-project/vllm
ones that match your environment. Please note that you may need to restart your runtime after installation.
'''
CPM_KERNELS_IMPORT_ERROR = '''{0} requires the cpm_kernels library but it was not found in your environment.
You can install it with pip: `pip install cpm_kernels`. Please note that you may need to restart your
runtime after installation.
'''
EINOPS_IMPORT_ERROR = '''{0} requires the einops library but it was not found in your environment.
You can install it with pip: `pip install einops`. Please note that you may need to restart
your runtime after installation.
'''
TRITON_IMPORT_ERROR = '''{0} requires the triton library but it was not found in your environment.
You can install it with pip: 'pip install \"git+https://github.com/openai/triton.git#egg=triton&subdirectory=python\"'.
Please note that you may need to restart your runtime after installation.
'''
DATASETS_IMPORT_ERROR = '''{0} requires the datasets library but it was not found in your environment.
You can install it with pip: `pip install datasets`. Please note that you may need to restart
your runtime after installation.
'''
PEFT_IMPORT_ERROR = '''{0} requires the peft library but it was not found in your environment.
You can install it with pip: `pip install peft`. Please note that you may need to restart
your runtime after installation.
'''
BITSANDBYTES_IMPORT_ERROR = '''{0} requires the bitsandbytes library but it was not found in your environment.
You can install it with pip: `pip install bitsandbytes`. Please note that you may need to restart
your runtime after installation.
'''
AUTOGPTQ_IMPORT_ERROR = '''{0} requires the auto-gptq library but it was not found in your environment.
You can install it with pip: `pip install auto-gptq`. Please note that you may need to restart
your runtime after installation.
'''
SENTENCEPIECE_IMPORT_ERROR = '''{0} requires the sentencepiece library but it was not found in your environment.
You can install it with pip: `pip install sentencepiece`. Please note that you may need to restart
your runtime after installation.
'''
XFORMERS_IMPORT_ERROR = '''{0} requires the xformers library but it was not found in your environment.
You can install it with pip: `pip install xformers`. Please note that you may need to restart
your runtime after installation.
'''
FAIRSCALE_IMPORT_ERROR = '''{0} requires the fairscale library but it was not found in your environment.
You can install it with pip: `pip install fairscale`. Please note that you may need to restart
your runtime after installation.
'''

BACKENDS_MAPPING: BackendOrderedDict = OrderedDict([('torch', (is_torch_available, PYTORCH_IMPORT_ERROR)), ('vllm', (is_vllm_available, VLLM_IMPORT_ERROR)),
                                                    ('cpm_kernels', (is_cpm_kernels_available, CPM_KERNELS_IMPORT_ERROR)), ('einops', (is_einops_available, EINOPS_IMPORT_ERROR)),
                                                    ('triton', (is_triton_available, TRITON_IMPORT_ERROR)), ('datasets', (is_datasets_available, DATASETS_IMPORT_ERROR)),
                                                    ('peft', (is_peft_available, PEFT_IMPORT_ERROR)), ('bitsandbytes', (is_bitsandbytes_available, BITSANDBYTES_IMPORT_ERROR)),
                                                    ('auto-gptq', (is_autogptq_available, AUTOGPTQ_IMPORT_ERROR)), ('sentencepiece', (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
                                                    ('xformers', (is_xformers_available, XFORMERS_IMPORT_ERROR)), ('fairscale', (is_fairscale_available, FAIRSCALE_IMPORT_ERROR))])

class DummyMetaclass(abc.ABCMeta):
  '''Metaclass for dummy object.

  It will raises ImportError generated by ``require_backends`` if users try to access attributes from given class.
  '''
  _backends: t.List[str]

  def __getattribute__(cls, key: str) -> t.Any:
    if key.startswith('_'): return super().__getattribute__(key)
    require_backends(cls, cls._backends)

def require_backends(o: t.Any, backends: t.MutableSequence[str]) -> None:
  if not isinstance(backends, (list, tuple)): backends = list(backends)
  name = o.__name__ if hasattr(o, '__name__') else o.__class__.__name__
  # Raise an error when vLLM is not available to consider PyTorch
  if 'vllm' in backends and 'torch' not in backends and is_torch_available() and not is_vllm_available():
    raise ImportError(VLLM_IMPORT_ERROR_WITH_PYTORCH.format(name))
  failed = [msg.format(name) for available, msg in (BACKENDS_MAPPING[backend] for backend in backends) if not available()]
  if failed: raise ImportError(''.join(failed))

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
    '''EnvVarMixin is a mixin class that returns the value extracted from environment variables.'''
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
    return LazyLoader(self.model_name, globals(), f'openllm.models.{self.model_name}')
