from typing import Any, Dict, Literal, Union

from transformers import AwqConfig, BitsAndBytesConfig, GPTQConfig

from openllm_core._typing_compat import LiteralQuantise, M, T, overload

from ._llm import LLM

QuantizationConfig = Union[BitsAndBytesConfig, GPTQConfig, AwqConfig]

@overload
def infer_quantisation_config(
  self: LLM[M, T], quantise: Literal['int8', 'int4'], **attrs: Any
) -> tuple[BitsAndBytesConfig, Dict[str, Any]]: ...
@overload
def infer_quantisation_config(
  self: LLM[M, T], quantise: Literal['gptq'], **attrs: Any
) -> tuple[GPTQConfig, Dict[str, Any]]: ...
@overload
def infer_quantisation_config(
  self: LLM[M, T], quantise: Literal['awq'], **attrs: Any
) -> tuple[AwqConfig, Dict[str, Any]]: ...
@overload
def infer_quantisation_config(
  self: LLM[M, T], quantise: LiteralQuantise, **attrs: Any
) -> tuple[QuantizationConfig, Dict[str, Any]]: ...
