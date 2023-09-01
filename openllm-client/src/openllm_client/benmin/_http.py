from __future__ import annotations
import asyncio
import functools
import logging
import time
import typing as t
import urllib.error

from urllib.parse import urlparse

import httpx
import orjson
import starlette.datastructures
import starlette.requests
import starlette.responses

import bentoml

from bentoml._internal.service.inference_api import InferenceAPI
from openllm_client.benmin import AsyncClient
from openllm_client.benmin import Client
from openllm_core.utils import ensure_exec_coro

logger = logging.getLogger(__name__)

class HttpClient(Client):
  @functools.cached_property
  def inner(self) -> httpx.Client:
    if not urlparse(self.server_url).netloc: raise ValueError(f'Invalid server url: {self.server_url}')
    return httpx.Client(base_url=self.server_url)

  @staticmethod
  def wait_until_server_ready(host: str, port: int, timeout: float = 30, check_interval: int = 1, **kwargs: t.Any) -> None:
    host = host if '://' in host else 'http://' + host
    logger.debug('Waiting for server @ `%s:%d` to be ready...', host, port)
    start = time.time()
    while time.time() - start < timeout:
      try:
        status = httpx.get(f'{host}:{port}/readyz').status_code
        if status == 200: break
        else: time.sleep(check_interval)
      except (httpx.ConnectError, urllib.error.URLError, ConnectionError):
        logger.debug('Server is not ready yet, retrying in %d seconds...', check_interval)
        time.sleep(check_interval)
    # Try once more and raise for exception
    try:
      httpx.get(f'{host}:{port}/readyz').raise_for_status()
    except httpx.HTTPStatusError as err:
      logger.error('Failed to wait until server ready: %s:%d', host, port)
      logger.error(err)
      raise

  def health(self) -> httpx.Response:
    return self.inner.get('/readyz')

  @classmethod
  def from_url(cls, url: str, **kwargs: t.Any) -> HttpClient:
    url = url if '://' in url else 'http://' + url
    resp = httpx.get(f'{url}/docs.json')
    if resp.status_code != 200:
      raise ValueError(f'Failed to get OpenAPI schema from the server: {resp.status_code} {resp.reason_phrase}:\n{resp.content.decode()}')
    _spec = orjson.loads(resp.content)

    reflection = bentoml.Service(_spec['info']['title'])

    for route, spec in _spec['paths'].items():
      for meth_spec in spec.values():
        if 'tags' in meth_spec and 'Service APIs' in meth_spec['tags']:
          if 'x-bentoml-io-descriptor' not in meth_spec['requestBody']:
            raise ValueError(f'Malformed BentoML spec received from BentoML server {url}')
          if 'x-bentoml-io-descriptor' not in meth_spec['responses']['200']:
            raise ValueError(f'Malformed BentoML spec received from BentoML server {url}')
          if 'x-bentoml-name' not in meth_spec:
            raise ValueError(f'Malformed BentoML spec received from BentoML server {url}')
          try:
            reflection.apis[meth_spec['x-bentoml-name']] = InferenceAPI[t.Any](None,
                                                                               bentoml.io.from_spec(meth_spec['requestBody']['x-bentoml-io-descriptor']),
                                                                               bentoml.io.from_spec(meth_spec['responses']['200']['x-bentoml-io-descriptor']),
                                                                               name=meth_spec['x-bentoml-name'],
                                                                               doc=meth_spec['description'],
                                                                               route=route.lstrip('/'))
          except Exception as e:
            logger.error('Failed to instantiate client for API %s: ', meth_spec['x-bentoml-name'], e)
    return cls(url, reflection)

  def _call(self, data: t.Any, /, *, _inference_api: InferenceAPI[t.Any], **kwargs: t.Any) -> t.Any:
    # All gRPC kwargs should be popped out.
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_grpc_')}
    if _inference_api.multi_input:
      if data is not None:
        raise ValueError(f"'{_inference_api.name}' takes multiple inputs, and thus required to pass as keyword arguments.")
      fake_resp = ensure_exec_coro(_inference_api.input.to_http_response(kwargs, None))
    else:
      fake_resp = ensure_exec_coro(_inference_api.input.to_http_response(data, None))

    # XXX: hack around StreamingResponse, since now we only have Text, for metadata so it is fine to do this.
    if isinstance(fake_resp, starlette.responses.StreamingResponse): body = None
    else: body = fake_resp.body

    resp = self.inner.post('/' + _inference_api.route if not _inference_api.route.startswith('/') else _inference_api.route,
                           data=body,
                           headers={'content-type': fake_resp.headers['content-type']},
                           timeout=self.timeout)
    if resp.status_code != 200: raise ValueError(f'Error while making request: {resp.status_code}: {resp.content!s}')
    fake_req = starlette.requests.Request(scope={'type': 'http'})
    headers = starlette.datastructures.Headers(headers=resp.headers)
    fake_req._body = resp.content
    # Request.headers sets a _headers variable. We will need to set this value to our fake request object.
    fake_req._headers = headers
    return ensure_exec_coro(_inference_api.output.from_http_request(fake_req))

class AsyncHttpClient(AsyncClient):
  @functools.cached_property
  def inner(self) -> httpx.AsyncClient:
    if not urlparse(self.server_url).netloc: raise ValueError(f'Invalid server url: {self.server_url}')
    return httpx.AsyncClient(base_url=self.server_url)

  @staticmethod
  async def wait_until_server_ready(host: str, port: int, timeout: float = 30, check_interval: int = 1, **kwargs: t.Any) -> None:
    host = host if '://' in host else 'http://' + host
    logger.debug('Waiting for server @ `%s:%d` to be ready...', host, port)
    start = time.time()
    while time.time() - start < timeout:
      try:
        async with httpx.AsyncClient(base_url=f'{host}:{port}') as sess:
          resp = await sess.get('/readyz')
          if resp.status_code == 200: break
          else: await asyncio.sleep(check_interval)
      except (httpx.ConnectError, urllib.error.URLError, ConnectionError):
        logger.debug('Server is not ready yet, retrying in %d seconds...', check_interval)
        await asyncio.sleep(check_interval)
    # Try once more and raise for exception
    async with httpx.AsyncClient(base_url=f'{host}:{port}') as sess:
      resp = await sess.get('/readyz')
      if resp.status_code != 200:
        raise TimeoutError(f'Timeout while waiting for server @ `{host}:{port}` to be ready: {resp.status_code}: {resp.content!s}')

  async def health(self) -> httpx.Response:
    return await self.inner.get('/readyz')

  @classmethod
  async def from_url(cls, url: str, **kwargs: t.Any) -> AsyncHttpClient:
    url = url if '://' in url else 'http://' + url
    async with httpx.AsyncClient(base_url=url) as session:
      resp = await session.get('/docs.json')
      if resp.status_code != 200:
        raise ValueError(f'Failed to get OpenAPI schema from the server: {resp.status_code} {resp.reason_phrase}:\n{(await resp.aread()).decode()}')
      _spec = orjson.loads(await resp.aread())

    reflection = bentoml.Service(_spec['info']['title'])

    for route, spec in _spec['paths'].items():
      for meth_spec in spec.values():
        if 'tags' in meth_spec and 'Service APIs' in meth_spec['tags']:
          if 'x-bentoml-io-descriptor' not in meth_spec['requestBody']:
            raise ValueError(f'Malformed BentoML spec received from BentoML server {url}')
          if 'x-bentoml-io-descriptor' not in meth_spec['responses']['200']:
            raise ValueError(f'Malformed BentoML spec received from BentoML server {url}')
          if 'x-bentoml-name' not in meth_spec:
            raise ValueError(f'Malformed BentoML spec received from BentoML server {url}')
          try:
            reflection.apis[meth_spec['x-bentoml-name']] = InferenceAPI[t.Any](None,
                                                                               bentoml.io.from_spec(meth_spec['requestBody']['x-bentoml-io-descriptor']),
                                                                               bentoml.io.from_spec(meth_spec['responses']['200']['x-bentoml-io-descriptor']),
                                                                               name=meth_spec['x-bentoml-name'],
                                                                               doc=meth_spec['description'],
                                                                               route=route.lstrip('/'))
          except ValueError as e:
            logger.error('Failed to instantiate client for API %s: ', meth_spec['x-bentoml-name'], e)
    return cls(url, reflection)

  async def _call(self, data: t.Any, /, *, _inference_api: InferenceAPI[t.Any], **kwargs: t.Any) -> t.Any:
    # All gRPC kwargs should be popped out.
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_grpc_')}
    if _inference_api.multi_input:
      if data is not None:
        raise ValueError(f"'{_inference_api.name}' takes multiple inputs, and thus required to pass as keyword arguments.")
      fake_resp = await _inference_api.input.to_http_response(kwargs, None)
    else:
      fake_resp = await _inference_api.input.to_http_response(data, None)

    # XXX: hack around StreamingResponse, since now we only have Text, for metadata so it is fine to do this.
    if isinstance(fake_resp, starlette.responses.StreamingResponse): body = None
    else: body = t.cast(t.Any, fake_resp.body)

    resp = await self.inner.post('/' + _inference_api.route if not _inference_api.route.startswith('/') else _inference_api.route,
                                 data=body,
                                 headers={'content-type': fake_resp.headers['content-type']},
                                 timeout=self.timeout)
    if resp.status_code != 200: raise ValueError(f'Error making request: {resp.status_code}: {(await resp.aread())!s}')
    fake_req = starlette.requests.Request(scope={'type': 'http'})
    headers = starlette.datastructures.Headers(headers=resp.headers)
    fake_req._body = resp.content
    # Request.headers sets a _headers variable. We will need to set this value to our fake request object.
    fake_req._headers = headers
    return await _inference_api.output.from_http_request(fake_req)
