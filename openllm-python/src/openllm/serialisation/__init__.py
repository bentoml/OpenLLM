from __future__ import annotations
import importlib, typing as t
from openllm_core._typing_compat import M, ParamSpec, T, TypeGuard, Concatenate
from openllm_core.exceptions import OpenLLMException

if t.TYPE_CHECKING:
  from bentoml import Model
  from .._llm import LLM

P = ParamSpec('P')


def load_tokenizer(llm: LLM[M, T], **tokenizer_attrs: t.Any) -> TypeGuard[T]:
  import cloudpickle, fs, transformers
  from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME
  from .transformers._helpers import process_config

  tokenizer_attrs = {**llm.llm_parameters[-1], **tokenizer_attrs}

  config, *_ = process_config(llm.bentomodel.path, llm.trust_remote_code)

  bentomodel_fs = fs.open_fs(llm.bentomodel.path)
  if bentomodel_fs.isfile(CUSTOM_OBJECTS_FILENAME):
    with bentomodel_fs.open(CUSTOM_OBJECTS_FILENAME, 'rb') as cofile:
      try:
        tokenizer = cloudpickle.load(cofile)['tokenizer']
      except KeyError:
        raise OpenLLMException(
          "Bento model does not have tokenizer. Make sure to save the tokenizer within the model via 'custom_objects'. "
          'For example: "bentoml.transformers.save_model(..., custom_objects={\'tokenizer\': tokenizer})"'
        ) from None
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(bentomodel_fs.getsyspath('/'), trust_remote_code=llm.trust_remote_code, **tokenizer_attrs)

  if tokenizer.pad_token_id is None:
    if config.pad_token_id is not None:
      tokenizer.pad_token_id = config.pad_token_id
    elif config.eos_token_id is not None:
      tokenizer.pad_token_id = config.eos_token_id
    elif tokenizer.eos_token_id is not None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
  return tokenizer


def _make_dispatch_function(fn: str) -> t.Callable[Concatenate[LLM[M, T], P], TypeGuard[M | T | Model]]:
  def caller(llm: LLM[M, T], *args: P.args, **kwargs: P.kwargs) -> TypeGuard[M | T | Model]:
    """Generic function dispatch to correct serialisation submodules based on LLM runtime.

    > [!NOTE] See 'openllm.serialisation.transformers' if 'llm.__llm_backend__ in ("pt", "vllm")'

    > [!NOTE] See 'openllm.serialisation.ggml' if 'llm.__llm_backend__="ggml"'

    > [!NOTE] See 'openllm.serialisation.ctranslate' if 'llm.__llm_backend__="ctranslate"'
    """
    if llm.__llm_backend__ == 'ggml':
      serde = 'ggml'
    elif llm.__llm_backend__ == 'ctranslate':
      serde = 'ctranslate'
    elif llm.__llm_backend__ in {'pt', 'vllm'}:
      serde = 'transformers'
    else:
      raise OpenLLMException(f'Not supported backend {llm.__llm_backend__}')
    return getattr(importlib.import_module(f'.{serde}', 'openllm.serialisation'), fn)(llm, *args, **kwargs)

  return caller


_extras = ['get', 'import_model', 'load_model']
_import_structure = {'ggml', 'transformers', 'ctranslate', 'constants'}
__all__ = ['load_tokenizer', *_extras, *_import_structure]


def __dir__() -> t.Sequence[str]:
  return sorted(__all__)


def __getattr__(name: str) -> t.Any:
  if name == 'load_tokenizer':
    return load_tokenizer
  elif name in _import_structure:
    return importlib.import_module(f'.{name}', __name__)
  elif name in _extras:
    return _make_dispatch_function(name)
  else:
    raise AttributeError(f'{__name__} has no attribute {name}')
