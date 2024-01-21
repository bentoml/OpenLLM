"""Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, and vLLM.

Currently, GGML format is working in progress.
"""

from typing import Any, Optional, TypeVar
from bentoml import Model as _Model
from openllm import LLM as _LLM
from . import constants as constants, ggml as ggml, transformers as transformers, vllm as vllm

M = TypeVar('M')
T = TypeVar('T')

def prepare_model(model_id: str, /, *, bentomodel_tag: Optional[str] = ...) -> _Model: ...
def import_model(*args: Any, trust_remote_code: bool, **attrs: Any) -> _Model: ...
def load_model(llm: _LLM[M, Any], *args: Any, **attrs: Any) -> M: ...
def load_tokenizer(llm: _LLM[Any, T], **attrs: Any) -> T: ...
