import types
from contextlib import contextmanager
from typing import Iterator, Optional, Sequence, Tuple

import transformers

from bentoml import Model
from openllm_core._typing_compat import M, T

from .._llm import LLM

def get_hash(config: transformers.PretrainedConfig) -> str: ...
def patch_correct_tag(llm: LLM[M, T], config: transformers.PretrainedConfig, _revision: Optional[str] = ...) -> None: ...
@contextmanager
def save_model(
  llm: LLM[M, T],
  config: transformers.PretrainedConfig,
  safe_serialisation: bool,
  trust_remote_code: bool,
  module: str,
  external_modules: Sequence[types.ModuleType],
) -> Iterator[Tuple[Model, Sequence[types.ModuleType]]]: ...
