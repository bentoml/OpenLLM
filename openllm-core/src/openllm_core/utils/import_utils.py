import importlib, importlib.metadata, importlib.util, os

OPTIONAL_DEPENDENCIES = {'vllm', 'fine-tune', 'ggml', 'ctranslate', 'agents', 'openai', 'playground', 'gptq', 'grpc', 'awq'}
ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({'AUTO'})
USE_VLLM = os.getenv('USE_VLLM', 'AUTO').upper()


def _is_package_available(package: str) -> bool:
  _package_available = importlib.util.find_spec(package) is not None
  if _package_available:
    try:
      importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      _package_available = False
  return _package_available


_ctranslate_available = importlib.util.find_spec('ctranslate2') is not None
_vllm_available = importlib.util.find_spec('vllm') is not None
_grpc_available = importlib.util.find_spec('grpc') is not None
_autoawq_available = importlib.util.find_spec('awq') is not None
_triton_available = importlib.util.find_spec('triton') is not None
_torch_available = _is_package_available('torch')
_transformers_available = _is_package_available('transformers')
_bentoml_available = _is_package_available('bentoml')
_peft_available = _is_package_available('peft')
_bitsandbytes_available = _is_package_available('bitsandbytes')
_flash_attn_available = _is_package_available('flash_attn')
_jupyter_available = _is_package_available('jupyter')
_jupytext_available = _is_package_available('jupytext')
_notebook_available = _is_package_available('notebook')
_autogptq_available = _is_package_available('auto_gptq')


def is_triton_available() -> bool:
  return _triton_available


def is_ctranslate_available() -> bool:
  return _ctranslate_available


def is_bentoml_available() -> bool:
  return _bentoml_available  # needs this since openllm-core doesn't explicitly depends on bentoml


def is_transformers_available() -> bool:
  return _transformers_available  # needs this since openllm-core doesn't explicitly depends on transformers


def is_grpc_available() -> bool:
  return _grpc_available


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
  return _torch_available


def is_flash_attn_2_available() -> bool:
  return _flash_attn_available


def is_autoawq_available() -> bool:
  global _autoawq_available
  try:
    importlib.metadata.version('autoawq')
  except importlib.metadata.PackageNotFoundError:
    _autoawq_available = False
  return _autoawq_available


def is_vllm_available() -> bool:
  global _vllm_available
  if USE_VLLM in ENV_VARS_TRUE_AND_AUTO_VALUES or _vllm_available:
    try:
      importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
      _vllm_available = False
  return _vllm_available
