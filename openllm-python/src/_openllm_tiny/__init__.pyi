from typing import Optional, Any
import bentoml
from bentoml._internal.models.model import ModelInfo
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, TypedDict, NotRequired
from ._llm import Dtype, LLM as LLM

class _Metadata(TypedDict):
  model_id: str
  dtype: str
  _revision: str
  _local: bool
  serialisation: str
  architectures: str
  trust_remote_code: bool
  api_version: str
  openllm_version: str
  openllm_core_version: str
  openllm_client_version: str
  quantize: NotRequired[str]

class _Info(ModelInfo):
  metadata: _Metadata  # type: ignore[assignment]

class _Model(bentoml.Model):
  info: _Info

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
) -> _Model: ...
