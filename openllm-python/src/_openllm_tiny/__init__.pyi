import bentoml
from bentoml._internal.models.model import ModelInfo
from openllm_core._typing_compat import TypedDict, NotRequired
from ._llm import LLM as LLM

class _Metadata(TypedDict):
  model_id: str
  dtype: str
  _revision: str
  _local: bool
  serialisation: str
  architectures: str
  trust_remote_code: bool
  api_version: str
  llm_type: str
  openllm_version: str
  openllm_core_version: str
  openllm_client_version: str
  quantize: NotRequired[str]

class _Info(ModelInfo):
  metadata: _Metadata  # type: ignore[assignment]

class _Model(bentoml.Model):
  info: _Info
