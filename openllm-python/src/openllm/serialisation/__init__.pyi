"""Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, and vLLM.

Currently, GGML format is working in progress.
"""

from typing import Any
from bentoml import Model
from openllm import LLM
from openllm_core._typing_compat import M, T
from . import constants as constants, ggml as ggml, transformers as transformers

def load_tokenizer(llm: LLM[M, T], **attrs: Any) -> T:
  """Load the tokenizer from BentoML store.

  By default, it will try to find the bentomodel whether it is in store..
  If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
  """

def get(llm: LLM[M, T]) -> Model: ...
def import_model(llm: LLM[M, T], *args: Any, trust_remote_code: bool, **attrs: Any) -> Model: ...
def load_model(llm: LLM[M, T], *args: Any, **attrs: Any) -> M: ...
