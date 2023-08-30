"""This holds a simple client implementation, somewhat similar to `bentoml.client`.

This module is subjected to change and to be merged upstream to BentoML.

```python
import openllm_client

client = openllm_client.benmin.Client.from_url("http://localhost:3000")
```

The client implementation won't include a dynamic assignment of the service endpoints, rather this should be called
via `client.call` or `await client.call`.
"""
from __future__ import annotations
import typing as t

from abc import abstractmethod

import attr
import httpx

import bentoml

if t.TYPE_CHECKING:
  from bentoml._internal.service.inference_api import InferenceAPI

__all__ = ['Client', 'AsyncClient']

@attr.define(init=False)
class Client:
  server_url: str
  endpoints: t.List[str]
  svc: bentoml.Service
  timeout: int = attr.field(default=30)

  def __init__(self, server_url: str, svc: bentoml.Service, **kwargs: t.Any) -> None:
    if len(svc.apis) == 0: raise bentoml.exceptions.BentoMLException('No APIs was found while constructing clients.')
    self.__attrs_init__(server_url=server_url, endpoints=list(svc.apis), svc=svc)
    for it, val in kwargs.items():
      object.__setattr__(self, it, val)

  def call(self, bentoml_api_name: str, data: t.Any = None, **kwargs: t.Any) -> t.Any:
    return self._call(data, _inference_api=self.svc.apis[bentoml_api_name], **kwargs)

  @abstractmethod
  def _call(self, data: t.Any, /, *, _inference_api: InferenceAPI[t.Any], **kwargs: t.Any) -> t.Any:
    raise NotImplementedError

  @abstractmethod
  def health(self) -> t.Any:
    raise NotImplementedError

  @classmethod
  def from_url(cls, url: str, **kwargs: t.Any) -> Client:
    try:
      from ._http import HttpClient
      return HttpClient.from_url(url, **kwargs)
    except httpx.RemoteProtocolError:
      from ._grpc import GrpcClient
      return GrpcClient.from_url(url, **kwargs)
    except Exception as err:
      raise bentoml.exceptions.BentoMLException('Failed to create client from url: %s' % url) from err

  @staticmethod
  def wait_until_server_ready(host: str, port: int, timeout: float = 30, **kwargs: t.Any) -> None:
    try:
      from ._http import HttpClient
      return HttpClient.wait_until_server_ready(host, port, timeout, **kwargs)
    except httpx.RemoteProtocolError:
      from ._grpc import GrpcClient
      return GrpcClient.wait_until_server_ready(host, port, timeout, **kwargs)
    except Exception as err:
      raise bentoml.exceptions.BentoMLException('Failed to wait until server ready: %s:%d' % (host, port)) from err

@attr.define(init=False)
class AsyncClient:
  server_url: str
  endpoints: t.List[str]
  svc: bentoml.Service
  timeout: int = attr.field(default=30)

  def __init__(self, server_url: str, svc: bentoml.Service, **kwargs: t.Any) -> None:
    if len(svc.apis) == 0: raise bentoml.exceptions.BentoMLException('No APIs was found while constructing clients.')
    self.__attrs_init__(server_url=server_url, endpoints=list(svc.apis), svc=svc)
    for it, val in kwargs.items():
      object.__setattr__(self, it, val)

  async def call(self, bentoml_api_name: str, data: t.Any = None, **kwargs: t.Any) -> t.Any:
    return await self._call(data, _inference_api=self.svc.apis[bentoml_api_name], **kwargs)

  @abstractmethod
  async def _call(self, data: t.Any, /, *, _inference_api: InferenceAPI[t.Any], **kwargs: t.Any) -> t.Any:
    raise NotImplementedError

  @abstractmethod
  async def health(self) -> t.Any:
    raise NotImplementedError

  @classmethod
  async def from_url(cls, url: str, **kwargs: t.Any) -> AsyncClient:
    try:
      from ._http import AsyncHttpClient
      return await AsyncHttpClient.from_url(url, **kwargs)
    except httpx.RemoteProtocolError:
      from ._grpc import AsyncGrpcClient
      return await AsyncGrpcClient.from_url(url, **kwargs)
    except Exception as err:
      raise bentoml.exceptions.BentoMLException('Failed to create client from url: %s' % url) from err

  @staticmethod
  async def wait_until_server_ready(host: str, port: int, timeout: float = 30, **kwargs: t.Any) -> None:
    try:
      from ._http import AsyncHttpClient
      await AsyncHttpClient.wait_until_server_ready(host, port, timeout, **kwargs)
    except httpx.RemoteProtocolError:
      from ._grpc import AsyncGrpcClient
      await AsyncGrpcClient.wait_until_server_ready(host, port, timeout, **kwargs)
    except Exception as err:
      raise bentoml.exceptions.BentoMLException('Failed to wait until server ready: %s:%d' % (host, port)) from err
