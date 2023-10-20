'''Entrypoint for all third-party apps.

Currently support OpenAI compatible API.

Each module should implement the following API:

- `mount_to_svc(svc: bentoml.Service, llm_runner: openllm.LLMRunner) -> bentoml.Service: ...`
'''
from __future__ import annotations
import os
import typing as t

from openllm_core.utils import LazyModule

_import_structure: dict[str, list[str]] = {'openai': []}

if t.TYPE_CHECKING:
  from . import openai as openai

__lazy = LazyModule(__name__, os.path.abspath('__file__'), _import_structure)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
