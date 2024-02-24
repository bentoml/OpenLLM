from typing import Optional, Any
import bentoml
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation
from ._llm import Dtype, LLM as LLM

def prepare_model(
  model_id: str,
  /,
  *decls: Any,
  bentomodel_tag: Optional[str] = ...,
  bentomodel_version: Optional[str] = ...,
  quantize: Optional[LiteralQuantise] = ...,
  dtype: Dtype = ...,
  serialisation: LiteralSerialisation = ...,
  trust_remote_code: bool = ...,
  low_cpu_mem_usage: bool = ...,
  **attrs: Any,
) -> bentoml.Model: ...
