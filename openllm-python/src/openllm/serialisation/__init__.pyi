"""Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, and vLLM.

Currently, GGML format is working in progress.
"""

from typing import Any
from bentoml import Model as _Model
from openllm import LLM as _LLM
from . import constants as constants, ggml as ggml, transformers as transformers, vllm as vllm

def import_model(*args: Any, trust_remote_code: bool, **attrs: Any) -> _Model: ...
def load_model(llm: _LLM, *args: Any, **attrs: Any) -> Any: ...
def load_tokenizer(llm: _LLM, **attrs: Any) -> Any: ...
