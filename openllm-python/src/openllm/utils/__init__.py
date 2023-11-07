"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations
import typing as t

import openllm_core

if t.TYPE_CHECKING:
  import openllm

def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> dict[str, t.Any]:
  return {'backend': llm.__llm_backend__, 'framework': 'openllm', 'model_name': llm.config['model_name'], 'architecture': llm.config['architecture'], 'serialisation': llm._serialisation}

__all__ = ['generate_labels']

def __dir__() -> t.Sequence[str]:
  return sorted(__all__)

def __getattr__(it: str) -> t.Any:
  if hasattr(openllm_core.utils, it): return getattr(openllm_core.utils, it)
  else: raise AttributeError(f'module {__name__} has no attribute {it}')
