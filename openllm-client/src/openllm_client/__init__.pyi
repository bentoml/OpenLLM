from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union
from typing import overload

import attr as _attr

from ._schemas import Response as _Response
from ._schemas import StreamingResponse as _StreamingResponse

@_attr.define
class HTTPClient:
  address: str
  client_args: Dict[str, Any]
  @staticmethod
  def wait_until_server_ready(
    addr: str, timeout: float = ..., verify: bool = ..., check_interval: int = ..., **client_args: Any
  ) -> None: ...
  @overload
  def __init__(
    self, address: str, timeout: int = ..., verify: bool = ..., api_version: str = ..., **client_args: Any
  ) -> None: ...
  @overload
  def __init__(
    self, address: str = ..., timeout: int = ..., verify: bool = ..., api_version: str = ..., **client_args: Any
  ) -> None: ...
  @overload
  def __init__(
    self, address: None = ..., timeout: int = ..., verify: bool = ..., api_version: str = ..., **client_args: Any
  ) -> None: ...
  @property
  def is_ready(self) -> bool: ...
  def health(self) -> None: ...
  def query(self, prompt: str, **attrs: Any) -> _Response: ...
  def generate(
    self,
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = ...,
    stop: Optional[Union[str, List[str]]] = ...,
    adapter_name: Optional[str] = ...,
    timeout: Optional[int] = ...,
    verify: Optional[bool] = ...,
    **attrs: Any,
  ) -> _Response: ...
  def generate_stream(
    self,
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = ...,
    stop: Optional[Union[str, List[str]]] = ...,
    adapter_name: Optional[str] = ...,
    timeout: Optional[int] = ...,
    verify: Optional[bool] = ...,
    **attrs: Any,
  ) -> Iterator[_StreamingResponse]: ...

@_attr.define
class AsyncHTTPClient:
  address: str
  client_args: Dict[str, Any]
  @staticmethod
  async def wait_until_server_ready(
    addr: str, timeout: float = ..., verify: bool = ..., check_interval: int = ..., **client_args: Any
  ) -> None: ...
  @overload
  def __init__(
    self, address: str, timeout: int = ..., verify: bool = ..., api_version: str = ..., **client_args: Any
  ) -> None: ...
  @overload
  def __init__(
    self, address: str = ..., timeout: int = ..., verify: bool = ..., api_version: str = ..., **client_args: Any
  ) -> None: ...
  @overload
  def __init__(
    self, address: None = ..., timeout: int = ..., verify: bool = ..., api_version: str = ..., **client_args: Any
  ) -> None: ...
  @property
  def is_ready(self) -> bool: ...
  async def health(self) -> None: ...
  async def query(self, prompt: str, **attrs: Any) -> _Response: ...
  async def generate(
    self,
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = ...,
    stop: Optional[Union[str, List[str]]] = ...,
    adapter_name: Optional[str] = ...,
    timeout: Optional[int] = ...,
    verify: Optional[bool] = ...,
    **attrs: Any,
  ) -> _Response: ...
  async def generate_stream(
    self,
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = ...,
    stop: Optional[Union[str, List[str]]] = ...,
    adapter_name: Optional[str] = ...,
    timeout: Optional[int] = ...,
    verify: Optional[bool] = ...,
    **attrs: Any,
  ) -> AsyncGenerator[_StreamingResponse, Any]: ...
