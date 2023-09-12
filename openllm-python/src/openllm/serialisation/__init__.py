'''Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, Tensorflow and Flax.

Currently, GGML format is working in progress.
'''
from __future__ import annotations
import importlib
import typing as t

import cloudpickle
import fs

import openllm

from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME
from openllm_core._typing_compat import M
from openllm_core._typing_compat import ParamSpec
from openllm_core._typing_compat import T

if t.TYPE_CHECKING:
  import bentoml

  from . import constants as constants
  from . import ggml as ggml
  from . import transformers as transformers

P = ParamSpec('P')

def load_tokenizer(llm: openllm.LLM[t.Any, T], **tokenizer_attrs: t.Any) -> T:
  '''Load the tokenizer from BentoML store.

  By default, it will try to find the bentomodel whether it is in store..
  If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
  '''
  from .transformers._helpers import infer_tokenizers_from_llm
  from .transformers._helpers import process_config

  config, *_ = process_config(llm._bentomodel.path, llm.trust_remote_code)

  bentomodel_fs = fs.open_fs(llm._bentomodel.path)
  if bentomodel_fs.isfile(CUSTOM_OBJECTS_FILENAME):
    with bentomodel_fs.open(CUSTOM_OBJECTS_FILENAME, 'rb') as cofile:
      try:
        tokenizer = cloudpickle.load(t.cast('t.IO[bytes]', cofile))['tokenizer']
      except KeyError:
        raise openllm.exceptions.OpenLLMException("Bento model does not have tokenizer. Make sure to save the tokenizer within the model via 'custom_objects'. "
                                                  "For example: \"bentoml.transformers.save_model(..., custom_objects={'tokenizer': tokenizer})\"") from None
  else:
    tokenizer = infer_tokenizers_from_llm(llm).from_pretrained(bentomodel_fs.getsyspath('/'), trust_remote_code=llm.trust_remote_code, **tokenizer_attrs)

  if tokenizer.pad_token_id is None:
    if config.pad_token_id is not None: tokenizer.pad_token_id = config.pad_token_id
    elif config.eos_token_id is not None: tokenizer.pad_token_id = config.eos_token_id
    elif tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id
    else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  return tokenizer

class _Caller(t.Protocol[P]):
  def __call__(self, llm: openllm.LLM[M, T], *args: P.args, **kwargs: P.kwargs) -> t.Any:
    ...

_extras = ['get', 'import_model', 'load_model']

def _make_dispatch_function(fn: str) -> _Caller[P]:
  def caller(llm: openllm.LLM[M, T], *args: P.args, **kwargs: P.kwargs) -> t.Any:
    """Generic function dispatch to correct serialisation submodules based on LLM runtime.

    > [!NOTE] See 'openllm.serialisation.transformers' if 'llm.__llm_backend__ in ("pt", "tf", "flax", "vllm")'

    > [!NOTE] See 'openllm.serialisation.ggml' if 'llm.__llm_backend__="ggml"'
    """
    serde = 'transformers'
    if llm.__llm_backend__ == 'ggml': serde = 'ggml'
    return getattr(importlib.import_module(f'.{serde}', __name__), fn)(llm, *args, **kwargs)

  return caller

if t.TYPE_CHECKING:

  def get(llm: openllm.LLM[M, T], *args: t.Any, **kwargs: t.Any) -> bentoml.Model:
    ...

  def import_model(llm: openllm.LLM[M, T], *args: t.Any, **kwargs: t.Any) -> bentoml.Model:
    ...

  def load_model(llm: openllm.LLM[M, T], *args: t.Any, **kwargs: t.Any) -> M:
    ...

_import_structure: dict[str, list[str]] = {'ggml': [], 'transformers': [], 'constants': []}
__all__ = ['ggml', 'transformers', 'constants', 'load_tokenizer', *_extras]

def __dir__() -> list[str]:
  return sorted(__all__)

def __getattr__(name: str) -> t.Any:
  if name == 'load_tokenizer': return load_tokenizer
  elif name in _import_structure: return importlib.import_module(f'.{name}', __name__)
  elif name in _extras: return _make_dispatch_function(name)
  else: raise AttributeError(f'{__name__} has no attribute {name}')
