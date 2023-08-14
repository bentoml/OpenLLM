from __future__ import annotations
import asyncio, logging, typing as t
import orjson, openllm
from openllm._typing_compat import LiteralRuntime
from .base import BaseAsyncClient, BaseClient

if t.TYPE_CHECKING:
  from grpc_health.v1 import health_pb2
  from bentoml.grpc.v1.service_pb2 import Response

logger = logging.getLogger(__name__)

class GrpcClient(BaseClient["Response"], client_type="grpc"):
  def __init__(self, address: str, timeout: int = 30):
    self._host, self._port = address.split(":")
    super().__init__(address, timeout)
  def health(self) -> health_pb2.HealthCheckResponse: return asyncio.run(self._cached.health("bentoml.grpc.v1.BentoService"))
  @property
  def model_name(self) -> str:
    try: return self._metadata.json.struct_value.fields["model_name"].string_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def framework(self) -> LiteralRuntime:
    try:
      value = t.cast(LiteralRuntime, self._metadata.json.struct_value.fields["framework"].string_value)
      if value not in ("pt", "flax", "tf", "vllm"): raise KeyError
      return value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def timeout(self) -> int:
    try: return int(self._metadata.json.struct_value.fields["timeout"].number_value)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def model_id(self) -> str:
    try: return self._metadata.json.struct_value.fields["model_id"].string_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def configuration(self) -> dict[str, t.Any]:
    try: return orjson.loads(self._metadata.json.struct_value.fields["configuration"].string_value)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_embeddings(self) -> bool:
    try: return self._metadata.json.struct_value.fields["supports_embeddings"].bool_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_hf_agent(self) -> bool:
    try: return self._metadata.json.struct_value.fields["supports_hf_agent"].bool_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  def postprocess(self, result: Response | dict[str, t.Any]) -> openllm.GenerationOutput:
    from google.protobuf.json_format import MessageToDict
    if isinstance(result, dict): return openllm.GenerationOutput(**result)
    return openllm.GenerationOutput(**MessageToDict(result.json, preserving_proto_field_name=True))

class AsyncGrpcClient(BaseAsyncClient["Response"], client_type="grpc"):
  def __init__(self, address: str, timeout: int = 30):
    self._host, self._port = address.split(":")
    super().__init__(address, timeout)
  async def health(self) -> health_pb2.HealthCheckResponse: return await self._cached.health("bentoml.grpc.v1.BentoService")
  @property
  def model_name(self) -> str:
    try: return self._metadata.json.struct_value.fields["model_name"].string_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def framework(self) -> LiteralRuntime:
    try:
      value = t.cast(LiteralRuntime, self._metadata.json.struct_value.fields["framework"].string_value)
      if value not in ("pt", "flax", "tf", "vllm"): raise KeyError
      return value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def timeout(self) -> int:
    try: return int(self._metadata.json.struct_value.fields["timeout"].number_value)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def model_id(self) -> str:
    try: return self._metadata.json.struct_value.fields["model_id"].string_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def configuration(self) -> dict[str, t.Any]:
    try: return orjson.loads(self._metadata.json.struct_value.fields["configuration"].string_value)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_embeddings(self) -> bool:
    try: return self._metadata.json.struct_value.fields["supports_embeddings"].bool_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_hf_agent(self) -> bool:
    try: return self._metadata.json.struct_value.fields["supports_hf_agent"].bool_value
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  def postprocess(self, result: Response | dict[str, t.Any]) -> openllm.GenerationOutput:
    from google.protobuf.json_format import MessageToDict
    if isinstance(result, dict): return openllm.GenerationOutput(**result)
    return openllm.GenerationOutput(**MessageToDict(result.json, preserving_proto_field_name=True))
