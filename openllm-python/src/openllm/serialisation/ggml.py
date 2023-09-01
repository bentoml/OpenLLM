'''Serialisation related implementation for GGML-based implementation.

This requires ctransformers to be installed.
'''
from __future__ import annotations
import typing as t

if t.TYPE_CHECKING:
  import bentoml
  import openllm

  from openllm_core._typing_compat import M

_conversion_strategy = {'pt': 'ggml'}

def import_model(llm: openllm.LLM[t.Any, t.Any], *decls: t.Any, trust_remote_code: bool = True, **attrs: t.Any,) -> bentoml.Model:
  raise NotImplementedError('Currently work in progress.')

def get(llm: openllm.LLM[t.Any, t.Any], auto_import: bool = False) -> bentoml.Model:
  raise NotImplementedError('Currently work in progress.')

def load_model(llm: openllm.LLM[M, t.Any], *decls: t.Any, **attrs: t.Any) -> M:
  raise NotImplementedError('Currently work in progress.')
