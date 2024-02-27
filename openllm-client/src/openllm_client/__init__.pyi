from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Sequence, Union, overload

import attr as _attr

from ._schemas import MesssageParam as _MesssageParam, Response as _Response, StreamingResponse as _StreamingResponse

class _Helpers:
  def messages(self, messages: Sequence[_MesssageParam], add_generation_prompt: bool = ...) -> str: ...

class _AsyncHelpers:
  async def messages(self, messages: Sequence[_MesssageParam], add_generation_prompt: bool = ...) -> str: ...

@_attr.define
class HTTPClient:
  address: str
  helpers: _Helpers
  @overload
  def __init__(
    self, address: str, timeout: int = ..., verify: bool = ..., max_retries: int = ..., api_version: str = ...
  ) -> None: ...
  @overload
  def __init__(
    self, address: str = ..., timeout: int = ..., verify: bool = ..., max_retries: int = ..., api_version: str = ...
  ) -> None: ...
  @overload
  def __init__(
    self, address: None = ..., timeout: int = ..., verify: bool = ..., max_retries: int = ..., api_version: str = ...
  ) -> None: ...
  @property
  def is_ready(self) -> bool: ...
  def health(self) -> bool: ...
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
  def generate_iterator(
    self,
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = ...,
    stop: Optional[Union[str, List[str]]] = ...,
    adapter_name: Optional[str] = ...,
    timeout: Optional[int] = ...,
    verify: Optional[bool] = ...,
    **attrs: Any,
  ) -> Iterator[_Response]: ...
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
  helpers: _AsyncHelpers
  @overload
  def __init__(
    self, address: str, timeout: int = ..., verify: bool = ..., max_retries: int = ..., api_version: str = ...
  ) -> None: ...
  @overload
  def __init__(
    self, address: str = ..., timeout: int = ..., verify: bool = ..., max_retries: int = ..., api_version: str = ...
  ) -> None: ...
  @overload
  def __init__(
    self, address: None = ..., timeout: int = ..., verify: bool = ..., max_retries: int = ..., api_version: str = ...
  ) -> None: ...
  @property
  def is_ready(self) -> bool: ...
  async def health(self) -> bool: ...
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
  async def generate_iterator(
    self,
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = ...,
    stop: Optional[Union[str, List[str]]] = ...,
    adapter_name: Optional[str] = ...,
    timeout: Optional[int] = ...,
    verify: Optional[bool] = ...,
    **attrs: Any,
  ) -> Iterator[_Response]: ...
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
