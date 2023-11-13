"""Entrypoint for all third-party apps.

Currently support OpenAI compatible API.

Each module should implement the following API:

- `mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service: ...`
"""

from __future__ import annotations
import importlib
import typing as t

from openllm_core.utils import LazyModule

if t.TYPE_CHECKING:
  import bentoml
  import openllm


class IntegrationModule(t.Protocol):
  def mount_to_svc(self, svc: bentoml.Service, llm: openllm.LLM[t.Any, t.Any]) -> bentoml.Service: ...


_import_structure: dict[str, list[str]] = {'openai': [], 'hf': [], 'cohere': []}


def mount_entrypoints(svc: bentoml.Service, llm: openllm.LLM[t.Any, t.Any]) -> bentoml.Service:
  for module_name in _import_structure:
    module = t.cast(IntegrationModule, importlib.import_module(f'.{module_name}', __name__))
    svc = module.mount_to_svc(svc, llm)
  return svc


__lazy = LazyModule(
  __name__, globals()['__file__'], _import_structure, extra_objects={'mount_entrypoints': mount_entrypoints}
)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
