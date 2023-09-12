# mypy: disable-error-code="no-redef"
from __future__ import annotations
import functools
import logging
import time
import typing as t

import bentoml

from bentoml._internal.service.inference_api import InferenceAPI
from bentoml.grpc.utils import import_generated_stubs
from bentoml.grpc.utils import load_from_file
from openllm_client.benmin import AsyncClient
from openllm_client.benmin import Client
from openllm_core._typing_compat import NotRequired
from openllm_core._typing_compat import overload
from openllm_core.utils import ensure_exec_coro
from openllm_core.utils import is_grpc_available
from openllm_core.utils import is_grpc_health_available

if not is_grpc_available() or not is_grpc_health_available():
  raise ImportError("gRPC is required to use gRPC client. Install with 'pip install \"openllm-client[grpc]\"'.")
import grpc
import grpc_health.v1.health_pb2 as pb_health
import grpc_health.v1.health_pb2_grpc as services_health

from google.protobuf import json_format
from grpc import aio

pb, services = import_generated_stubs('v1')

if t.TYPE_CHECKING:
  from bentoml.grpc.v1.service_pb2 import ServiceMetadataResponse

logger = logging.getLogger(__name__)

class ClientCredentials(t.TypedDict):
  root_certificates: NotRequired[t.Union[bytes, str]]
  private_key: NotRequired[t.Union[bytes, str]]
  certificate_chain: NotRequired[t.Union[bytes, str]]

@overload
def dispatch_channel(server_url: str,
                     typ: t.Literal['async'],
                     ssl: bool = ...,
                     ssl_client_credentials: ClientCredentials | None = ...,
                     options: t.Any | None = ...,
                     compression: grpc.Compression | None = ...,
                     interceptors: t.Sequence[aio.ClientInterceptor] | None = ...) -> aio.Channel:
  ...

@overload
def dispatch_channel(server_url: str,
                     typ: t.Literal['sync'],
                     ssl: bool = ...,
                     ssl_client_credentials: ClientCredentials | None = ...,
                     options: t.Any | None = ...,
                     compression: grpc.Compression | None = ...,
                     interceptors: t.Sequence[aio.ClientInterceptor] | None = None) -> grpc.Channel:
  ...

def dispatch_channel(server_url: str,
                     typ: t.Literal['async', 'sync'] = 'sync',
                     ssl: bool = False,
                     ssl_client_credentials: ClientCredentials | None = None,
                     options: t.Any | None = None,
                     compression: grpc.Compression | None = None,
                     interceptors: t.Sequence[aio.ClientInterceptor] | None = None) -> aio.Channel | grpc.Channel:
  credentials = None
  if ssl:
    if ssl_client_credentials is None: raise RuntimeError("'ssl=True' requires 'ssl_client_credentials'")
    credentials = grpc.ssl_channel_credentials(**{k: load_from_file(v) if isinstance(v, str) else v for k, v in ssl_client_credentials.items()})

  if typ == 'async' and ssl:
    return aio.secure_channel(server_url, credentials=credentials, options=options, compression=compression, interceptors=interceptors)
  elif typ == 'async':
    return aio.insecure_channel(server_url, options=options, compression=compression, interceptors=interceptors)
  elif typ == 'sync' and ssl:
    return grpc.secure_channel(server_url, credentials=credentials, options=options, compression=compression)
  elif typ == 'sync':
    return grpc.insecure_channel(server_url, options=options, compression=compression)
  else:
    raise ValueError(f'Unknown type: {typ}')

class GrpcClient(Client):
  ssl: bool
  ssl_client_credentials: t.Optional[ClientCredentials]
  options: t.Any
  compression: t.Optional[grpc.Compression]

  def __init__(self,
               server_url: str,
               svc: bentoml.Service,  # gRPC specific options
               ssl: bool = False,
               options: t.Any | None = None,
               compression: grpc.Compression | None = None,
               ssl_client_credentials: ClientCredentials | None = None,
               **kwargs: t.Any) -> None:
    self.ssl, self.ssl_client_credentials, self.options, self.compression = ssl, ssl_client_credentials, options, compression
    super().__init__(server_url, svc, **kwargs)

  @functools.cached_property
  def inner(self) -> grpc.Channel:
    if self.ssl:
      if self.ssl_client_credentials is None: raise RuntimeError("'ssl=True' requires 'ssl_client_credentials'")
      credentials = grpc.ssl_channel_credentials(**{k: load_from_file(v) if isinstance(v, str) else v for k, v in self.ssl_client_credentials.items()})
      return grpc.secure_channel(self.server_url, credentials=credentials, options=self.options, compression=self.compression)
    return grpc.insecure_channel(self.server_url, options=self.options, compression=self.compression)

  @staticmethod
  def wait_until_server_ready(host: str, port: int, timeout: float = 30, check_interval: int = 1, **kwargs: t.Any) -> None:
    with dispatch_channel(f"{host.replace(r'localhost', '0.0.0.0')}:{port}",
                          typ='sync',
                          options=kwargs.get('options', None),
                          compression=kwargs.get('compression', None),
                          ssl=kwargs.get('ssl', False),
                          ssl_client_credentials=kwargs.get('ssl_client_credentials', None)) as channel:
      req = pb_health.HealthCheckRequest()
      req.service = 'bentoml.grpc.v1.BentoService'
      health_stub = services_health.HealthStub(channel)
      start_time = time.time()
      while time.time() - start_time < timeout:
        try:
          resp = health_stub.Check(req)
          if resp.status == pb_health.HealthCheckResponse.SERVING: break
          else: time.sleep(check_interval)
        except grpc.RpcError:
          logger.debug('Waiting for server to be ready...')
          time.sleep(check_interval)
      try:
        resp = health_stub.Check(req)
        if resp.status != pb_health.HealthCheckResponse.SERVING:
          raise TimeoutError(f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready.")
      except grpc.RpcError as err:
        logger.error('Caught RpcError while connecting to %s:%s:\n', host, port)
        logger.error(err)
        raise

  @classmethod
  def from_url(cls, url: str, **kwargs: t.Any) -> GrpcClient:
    with dispatch_channel(url.replace(r'localhost', '0.0.0.0'),
                          typ='sync',
                          options=kwargs.get('options', None),
                          compression=kwargs.get('compression', None),
                          ssl=kwargs.get('ssl', False),
                          ssl_client_credentials=kwargs.get('ssl_client_credentials', None)) as channel:
      metadata = t.cast(
          'ServiceMetadataResponse',
          channel.unary_unary('/bentoml.grpc.v1.BentoService/ServiceMetadata',
                              request_serializer=pb.ServiceMetadataRequest.SerializeToString,
                              response_deserializer=pb.ServiceMetadataResponse.FromString)(pb.ServiceMetadataRequest()))
    reflection = bentoml.Service(metadata.name)
    for api in metadata.apis:
      try:
        reflection.apis[api.name] = InferenceAPI[t.Any](None,
                                                        bentoml.io.from_spec({
                                                            'id': api.input.descriptor_id, 'args': json_format.MessageToDict(api.input.attributes).get('args', None)
                                                        }),
                                                        bentoml.io.from_spec({
                                                            'id': api.output.descriptor_id, 'args': json_format.MessageToDict(api.output.attributes).get('args', None)
                                                        }),
                                                        name=api.name,
                                                        doc=api.docs)
      except Exception as e:
        logger.error('Failed to instantiate client for API %s: ', api.name, e)
    return cls(url, reflection, **kwargs)

  def health(self) -> t.Any:
    return services_health.HealthStub(self.inner).Check(pb_health.HealthCheckRequest(service=''))

  def _call(self, data: t.Any, /, *, _inference_api: InferenceAPI[t.Any], **kwargs: t.Any) -> t.Any:
    channel_kwargs = {k: kwargs.pop(f'_grpc_channel_{k}', None) for k in {'timeout', 'metadata', 'credentials', 'wait_for_ready', 'compression'}}
    if _inference_api.multi_input:
      if data is not None:
        raise ValueError(f"'{_inference_api.name}' takes multiple inputs, and thus required to pass as keyword arguments.")
      fake_resp = ensure_exec_coro(_inference_api.input.to_proto(kwargs))
    else:
      fake_resp = ensure_exec_coro(_inference_api.input.to_proto(data))
    api_fn = {v: k for k, v in self.svc.apis.items()}
    stubs = services.BentoServiceStub(self.inner)
    proto = stubs.Call(pb.Request(**{'api_name': api_fn[_inference_api], _inference_api.input.proto_fields[0]: fake_resp}), **channel_kwargs)
    return ensure_exec_coro(_inference_api.output.from_proto(getattr(proto, proto.WhichOneof('content'))))

class AsyncGrpcClient(AsyncClient):
  ssl: bool
  ssl_client_credentials: t.Optional[ClientCredentials]
  options: aio.ChannelArgumentType
  interceptors: t.Optional[t.Sequence[aio.ClientInterceptor]]
  compression: t.Optional[grpc.Compression]

  def __init__(self,
               server_url: str,
               svc: bentoml.Service,  # gRPC specific options
               ssl: bool = False,
               options: aio.ChannelArgumentType | None = None,
               interceptors: t.Sequence[aio.ClientInterceptor] | None = None,
               compression: grpc.Compression | None = None,
               ssl_client_credentials: ClientCredentials | None = None,
               **kwargs: t.Any) -> None:
    self.ssl, self.ssl_client_credentials, self.options, self.interceptors, self.compression = ssl, ssl_client_credentials, options, interceptors, compression
    super().__init__(server_url, svc, **kwargs)

  @functools.cached_property
  def inner(self) -> aio.Channel:
    if self.ssl:
      if self.ssl_client_credentials is None: raise RuntimeError("'ssl=True' requires 'ssl_client_credentials'")
      credentials = grpc.ssl_channel_credentials(**{k: load_from_file(v) if isinstance(v, str) else v for k, v in self.ssl_client_credentials.items()})
      return aio.secure_channel(self.server_url, credentials=credentials, options=self.options, compression=self.compression, interceptors=self.interceptors)
    return aio.insecure_channel(self.server_url, options=self.options, compression=self.compression, interceptors=self.interceptors)

  @staticmethod
  async def wait_until_server_ready(host: str, port: int, timeout: float = 30, check_interval: int = 1, **kwargs: t.Any) -> None:
    async with dispatch_channel(f"{host.replace(r'localhost', '0.0.0.0')}:{port}",
                                typ='async',
                                options=kwargs.get('options', None),
                                compression=kwargs.get('compression', None),
                                ssl=kwargs.get('ssl', False),
                                ssl_client_credentials=kwargs.get('ssl_client_credentials', None)) as channel:
      req = pb_health.HealthCheckRequest()
      req.service = 'bentoml.grpc.v1.BentoService'
      health_stub = services_health.HealthStub(channel)
      start_time = time.time()
      while time.time() - start_time < timeout:
        try:
          resp = health_stub.Check(req)
          if resp.status == pb_health.HealthCheckResponse.SERVING: break
          else: time.sleep(check_interval)
        except grpc.RpcError:
          logger.debug('Waiting for server to be ready...')
          time.sleep(check_interval)
      try:
        resp = health_stub.Check(req)
        if resp.status != pb_health.HealthCheckResponse.SERVING:
          raise TimeoutError(f"Timed out waiting {timeout} seconds for server at '{host}:{port}' to be ready.")
      except grpc.RpcError as err:
        logger.error('Caught RpcError while connecting to %s:%s:\n', host, port)
        logger.error(err)
        raise

  @classmethod
  async def from_url(cls, url: str, **kwargs: t.Any) -> AsyncGrpcClient:
    async with dispatch_channel(url.replace(r'localhost', '0.0.0.0'),
                                typ='async',
                                options=kwargs.get('options', None),
                                compression=kwargs.get('compression', None),
                                ssl=kwargs.get('ssl', False),
                                ssl_client_credentials=kwargs.get('ssl_client_credentials', None),
                                interceptors=kwargs.get('interceptors', None)) as channel:
      metadata = t.cast(
          'ServiceMetadataResponse',
          channel.unary_unary('/bentoml.grpc.v1.BentoService/ServiceMetadata',
                              request_serializer=pb.ServiceMetadataRequest.SerializeToString,
                              response_deserializer=pb.ServiceMetadataResponse.FromString)(pb.ServiceMetadataRequest()))
    reflection = bentoml.Service(metadata.name)
    for api in metadata.apis:
      try:
        reflection.apis[api.name] = InferenceAPI[t.Any](None,
                                                        bentoml.io.from_spec({
                                                            'id': api.input.descriptor_id, 'args': json_format.MessageToDict(api.input.attributes).get('args', None)
                                                        }),
                                                        bentoml.io.from_spec({
                                                            'id': api.output.descriptor_id, 'args': json_format.MessageToDict(api.output.attributes).get('args', None)
                                                        }),
                                                        name=api.name,
                                                        doc=api.docs)
      except Exception as e:
        logger.error('Failed to instantiate client for API %s: ', api.name, e)
    return cls(url, reflection, **kwargs)

  async def health(self) -> t.Any:
    return await services_health.HealthStub(self.inner).Check(pb_health.HealthCheckRequest(service=''))

  async def _call(self, data: t.Any, /, *, _inference_api: InferenceAPI[t.Any], **kwargs: t.Any) -> t.Any:
    channel_kwargs = {k: kwargs.pop(f'_grpc_channel_{k}', None) for k in {'timeout', 'metadata', 'credentials', 'wait_for_ready', 'compression'}}
    state = self.inner.get_state(try_to_connect=True)
    if state != grpc.ChannelConnectivity.READY: await self.inner.channel_ready()
    if _inference_api.multi_input:
      if data is not None:
        raise ValueError(f"'{_inference_api.name}' takes multiple inputs, and thus required to pass as keyword arguments.")
      fake_resp = await _inference_api.input.to_proto(kwargs)
    else:
      fake_resp = await _inference_api.input.to_proto(data)
    api_fn = {v: k for k, v in self.svc.apis.items()}
    async with self.inner:
      stubs = services.BentoServiceStub(self.inner)
      proto = await stubs.Call(pb.Request(**{'api_name': api_fn[_inference_api], _inference_api.input.proto_fields[0]: fake_resp}), **channel_kwargs)
    return await _inference_api.output.from_proto(getattr(proto, proto.WhichOneof('content')))
