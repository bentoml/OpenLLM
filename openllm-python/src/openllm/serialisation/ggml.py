'''Serialisation related implementation for GGML-based implementation.

This requires ctransformers to be installed.
'''
from __future__ import annotations
import typing as t

import bentoml
import openllm
if t.TYPE_CHECKING: from openllm_core._typing_compat import M

_conversion_strategy = {'pt': 'ggml'}

def import_model(llm: openllm.LLM[t.Any, t.Any], *decls: t.Any, trust_remote_code: bool = True, **attrs: t.Any,) -> bentoml.Model:
  raise NotImplementedError('Currently work in progress.')

def get(llm: openllm.LLM[t.Any, t.Any], auto_import: bool = False) -> bentoml.Model:
  '''Return an instance of ``bentoml.Model`` from given LLM instance.

  By default, it will try to check the model in the local store.
  If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

  Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
  '''
  try:
    model = bentoml.models.get(llm.tag)
    if model.info.module not in ('openllm.serialisation.ggml', __name__):
      raise bentoml.exceptions.NotFound(f"Model {model.tag} was saved with module {model.info.module}, not loading with 'openllm.serialisation.transformers'.")
    if 'runtime' in model.info.labels and model.info.labels['runtime'] != llm.runtime:
      raise openllm.exceptions.OpenLLMException(f"Model {model.tag} was saved with runtime {model.info.labels['runtime']}, not loading with {llm.runtime}.")
    return model
  except bentoml.exceptions.NotFound:
    if auto_import:
      return import_model(llm, trust_remote_code=llm.__llm_trust_remote_code__)
    raise

def load_model(llm: openllm.LLM[M, t.Any], *decls: t.Any, **attrs: t.Any) -> M:
  raise NotImplementedError('Currently work in progress.')

def save_pretrained(llm: openllm.LLM[t.Any, t.Any], save_directory: str, **attrs: t.Any) -> None:
  raise NotImplementedError('Currently work in progress.')
