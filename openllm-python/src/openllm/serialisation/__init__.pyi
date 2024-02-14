"""Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, and vLLM.

Currently, GGML format is working in progress.
"""

from typing import Any, Optional, TypeVar
from bentoml import Model as _Model
from openllm import LLM as _LLM
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, LiteralBackend
from .._quantisation import QuantizationConfig
from .._llm import Dtype
from . import constants as constants, ggml as ggml, transformers as transformers, vllm as vllm

M = TypeVar('M')
T = TypeVar('T')

def prepare_model(
  model_id: str,
  /,
  *decls: Any,
  bentomodel_tag: Optional[str] = ...,
  bentomodel_version: Optional[str] = ...,
  quantize: Optional[LiteralQuantise] = ...,
  quantization_config: Optional[QuantizationConfig] = ...,
  backend: Optional[LiteralBackend] = ...,
  dtype: Dtype = ...,
  serialisation: LiteralSerialisation = ...,
  trust_remote_code: bool = ...,
  low_cpu_mem_usage: bool = ...,
  **attrs: Any,
) -> _Model: ...
def import_model(*args: Any, trust_remote_code: bool, **attrs: Any) -> _Model: ...
def load_model(llm: _LLM[M, Any], *args: Any, **attrs: Any) -> M: ...
def load_tokenizer(llm: _LLM[Any, T], **attrs: Any) -> T: ...
