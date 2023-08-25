"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations
import typing as t, openllm_core
from . import (dummy_flax_objects as dummy_flax_objects, dummy_pt_objects as dummy_pt_objects, dummy_tf_objects as dummy_tf_objects, dummy_vllm_objects as dummy_vllm_objects,)

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import LiteralRuntime
  import openllm
def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> dict[str, t.Any]:
  return {'runtime': llm.runtime, 'framework': 'openllm', 'model_name': llm.config['model_name'], 'architecture': llm.config['architecture'], 'serialisation_format': llm._serialisation_format}
def infer_auto_class(implementation: LiteralRuntime) -> type[openllm.AutoLLM | openllm.AutoTFLLM | openllm.AutoFlaxLLM | openllm.AutoVLLM]:
  import openllm
  if implementation == 'tf': return openllm.AutoTFLLM
  elif implementation == 'flax': return openllm.AutoFlaxLLM
  elif implementation == 'pt': return openllm.AutoLLM
  elif implementation == 'vllm': return openllm.AutoVLLM
  else: raise RuntimeError(f"Unknown implementation: {implementation} (supported: 'pt', 'flax', 'tf', 'vllm')")
__all__ = ['generate_labels', 'infer_auto_class', 'dummy_flax_objects', 'dummy_pt_objects', 'dummy_tf_objects', 'dummy_vllm_objects']
def __dir__() -> t.Sequence[str]:
  return sorted(__all__)
def __getattr__(it: str) -> t.Any:
  if hasattr(openllm_core.utils, it): return getattr(openllm_core.utils, it)
  else: raise AttributeError(f'module {__name__} has no attribute {it}')
