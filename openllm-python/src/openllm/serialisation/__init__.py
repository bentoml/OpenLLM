from __future__ import annotations
import importlib

import cloudpickle
import fs

from openllm_core._typing_compat import ParamSpec
from openllm_core.exceptions import OpenLLMException

P = ParamSpec('P')


def load_tokenizer(llm, **tokenizer_attrs):
  """Load the tokenizer from BentoML store.

  By default, it will try to find the bentomodel whether it is in store..
  If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
  """
  from transformers import AutoTokenizer

  tokenizer_attrs = {**llm.llm_parameters[-1], **tokenizer_attrs}
  from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME

  from .transformers._helpers import process_config

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
    tokenizer = AutoTokenizer.from_pretrained(
      bentomodel_fs.getsyspath('/'), trust_remote_code=llm.trust_remote_code, **tokenizer_attrs
    )

  if tokenizer.pad_token_id is None:
    if config.pad_token_id is not None:
      tokenizer.pad_token_id = config.pad_token_id
    elif config.eos_token_id is not None:
      tokenizer.pad_token_id = config.eos_token_id
    elif tokenizer.eos_token_id is not None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
      tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  return tokenizer


_extras = ['get', 'import_model', 'load_model']


def _make_dispatch_function(fn):
  def caller(llm, *args, **kwargs):
    """Generic function dispatch to correct serialisation submodules based on LLM runtime.

    > [!NOTE] See 'openllm.serialisation.transformers' if 'llm.__llm_backend__ in ("pt", "vllm")'

    > [!NOTE] See 'openllm.serialisation.ggml' if 'llm.__llm_backend__="ggml"'
    """
    serde = 'transformers'
    if llm.__llm_backend__ == 'ggml':
      serde = 'ggml'
    return getattr(importlib.import_module(f'.{serde}', __name__), fn)(llm, *args, **kwargs)

  return caller


_import_structure: dict[str, list[str]] = {'ggml': [], 'transformers': [], 'constants': []}
__all__ = ['ggml', 'transformers', 'constants', 'load_tokenizer', *_extras]


def __dir__():
  return sorted(__all__)


def __getattr__(name):
  if name == 'load_tokenizer':
    return load_tokenizer
  elif name in _import_structure:
    return importlib.import_module(f'.{name}', __name__)
  elif name in _extras:
    return _make_dispatch_function(name)
  else:
    raise AttributeError(f'{__name__} has no attribute {name}')
