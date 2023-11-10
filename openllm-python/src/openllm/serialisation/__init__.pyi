"""Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, and vLLM.

Currently, GGML format is working in progress.
"""

from typing import Any

from bentoml import Model
from openllm import LLM
from openllm_core._typing_compat import M
from openllm_core._typing_compat import T

from . import constants as constants
from . import ggml as ggml
from . import transformers as transformers

def load_tokenizer(llm: LLM[M, T], **attrs: Any) -> T: ...
def get(llm: LLM[M, T]) -> Model: ...
def import_model(llm: LLM[M, T], *args: Any, trust_remote_code: bool, **attrs: Any) -> Model: ...
def load_model(llm: LLM[M, T], *args: Any, **attrs: Any) -> M: ...
