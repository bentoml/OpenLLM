"""Entrypoint for all third-party apps.

Currently support OpenAI compatible API.

Each module should implement the following API:

- `mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service: ...`
"""
from __future__ import annotations
import typing as t

from openllm_core.utils import LazyModule

from . import hf as hf
from . import openai as openai

if t.TYPE_CHECKING:
  import bentoml
  import openllm

_import_structure: dict[str, list[str]] = {'openai': [], 'hf': []}

def mount_entrypoints(svc: bentoml.Service, llm: openllm.LLM[t.Any, t.Any]) -> bentoml.Service:
  return openai.mount_to_svc(hf.mount_to_svc(svc, llm), llm)

__lazy = LazyModule(__name__, globals()['__file__'], _import_structure, extra_objects={'mount_entrypoints': mount_entrypoints})
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
