"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations
import typing as t

import openllm_core

from . import dummy_pt_objects as dummy_pt_objects
from . import dummy_vllm_objects as dummy_vllm_objects

if t.TYPE_CHECKING:
  import openllm

  from openllm_core._typing_compat import LiteralBackend

def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> dict[str, t.Any]:
  return {'backend': llm.__llm_backend__, 'framework': 'openllm', 'model_name': llm.config['model_name'], 'architecture': llm.config['architecture'], 'serialisation': llm._serialisation}

def infer_auto_class(backend: LiteralBackend) -> type[openllm.AutoLLM | openllm.AutoVLLM]:
  import openllm
  if backend == 'pt': return openllm.AutoLLM
  elif backend == 'vllm': return openllm.AutoVLLM
  else: raise RuntimeError(f"Unknown backend: {backend} (supported: 'pt', 'vllm')")

__all__ = ['generate_labels', 'dummy_pt_objects', 'dummy_vllm_objects']

def __dir__() -> t.Sequence[str]: return sorted(__all__)

def __getattr__(it: str) -> t.Any:
  if hasattr(openllm_core.utils, it): return getattr(openllm_core.utils, it)
  else: raise AttributeError(f'module {__name__} has no attribute {it}')
