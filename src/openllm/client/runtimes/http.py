from __future__ import annotations
import logging, typing as t
from urllib.parse import urljoin, urlparse
import httpx, orjson, openllm
from .base import BaseAsyncClient, BaseClient, in_async_context
from openllm._typing_compat import DictStrAny, LiteralRuntime

logger = logging.getLogger(__name__)
def process_address(self: AsyncHTTPClient | HTTPClient, address: str) -> None:
  address = address if "://" in address else "http://" + address
  parsed = urlparse(address)
  self._host, *_port = parsed.netloc.split(":")
  if len(_port) == 0: self._port = "80" if parsed.scheme == "http" else "443"
  else: self._port = next(iter(_port))

class HTTPClient(BaseClient[DictStrAny]):
  def __init__(self, address: str, timeout: int = 30):
    process_address(self, address)
    super().__init__(address, timeout)

  def health(self) -> t.Any: return self._cached.health()
  def embed(self, prompt: t.Sequence[str] | str) -> openllm.EmbeddingsOutput:
    if not self.supports_embeddings: raise ValueError("This model does not support embeddings.")
    if isinstance(prompt, str): prompt = [prompt]
    result = httpx.post(urljoin(self._address, f"/{self._api_version}/embeddings"), json=list(prompt), timeout=self.timeout).json() if in_async_context() else self.call("embeddings", list(prompt))
    return openllm.EmbeddingsOutput(**result)

  @property
  def model_name(self) -> str:
    try: return self._metadata["model_name"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def model_id(self) -> str:
    try: return self._metadata["model_name"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def framework(self) -> LiteralRuntime:
    try: return self._metadata["framework"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def timeout(self) -> int:
    try: return self._metadata["timeout"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def configuration(self) -> dict[str, t.Any]:
    try: return orjson.loads(self._metadata["configuration"])
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_embeddings(self) -> bool:
    try: return self._metadata.get("supports_embeddings", False)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_hf_agent(self) -> bool:
    try: return self._metadata.get("supports_hf_agent", False)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  def postprocess(self, result: dict[str, t.Any]) -> openllm.GenerationOutput: return openllm.GenerationOutput(**result)

class AsyncHTTPClient(BaseAsyncClient[DictStrAny]):
  def __init__(self, address: str, timeout: int = 30):
    process_address(self, address)
    super().__init__(address, timeout)

  async def health(self) -> t.Any: return await self._cached.async_health()
  async def embed(self, prompt: t.Sequence[str] | str) -> openllm.EmbeddingsOutput:
    if not self.supports_embeddings: raise ValueError("This model does not support embeddings.")
    if isinstance(prompt, str): prompt = [prompt]
    res = await self.acall("embeddings", list(prompt))
    return openllm.EmbeddingsOutput(**res)

  @property
  def model_name(self) -> str:
    try: return self._metadata["model_name"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def model_id(self) -> str:
    try: return self._metadata["model_name"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def framework(self) -> LiteralRuntime:
    try: return self._metadata["framework"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def timeout(self) -> int:
    try: return self._metadata["timeout"]
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def configuration(self) -> dict[str, t.Any]:
    try: return orjson.loads(self._metadata["configuration"])
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_embeddings(self) -> bool:
    try: return self._metadata.get("supports_embeddings", False)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  @property
  def supports_hf_agent(self) -> bool:
    try: return self._metadata.get("supports_hf_agent", False)
    except KeyError: raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None
  def postprocess(self, result: dict[str, t.Any]) -> openllm.GenerationOutput: return openllm.GenerationOutput(**result)
