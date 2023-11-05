from __future__ import annotations
import functools
import inspect
import typing as t

import attr

from starlette.routing import BaseRoute
from starlette.routing import Host
from starlette.routing import Mount
from starlette.routing import Route
from starlette.schemas import EndpointInfo
from starlette.schemas import SchemaGenerator

from openllm_core.utils import first_not_none

if t.TYPE_CHECKING:
  from attr import AttrsInstance

  import bentoml

OPENAPI_VERSION = '3.0.2'
API_VERSION = '1.0'

class OpenLLMSchemaGenerator(SchemaGenerator):
  def get_endpoints(self, routes: list[BaseRoute]) -> list[EndpointInfo]:
    endpoints_info: list[EndpointInfo] = []
    for route in routes:
      if isinstance(route, (Mount, Host)):
        routes = route.routes or []
        path = self._remove_converter(route.path) if isinstance(route, Mount) else ''
        sub_endpoints = [EndpointInfo(path=f'{path}{sub_endpoint.path}', http_method=sub_endpoint.http_method, func=sub_endpoint.func) for sub_endpoint in self.get_endpoints(routes)]
        endpoints_info.extend(sub_endpoints)
      elif not isinstance(route, Route) or not route.include_in_schema:
        continue
      elif inspect.isfunction(route.endpoint) or inspect.ismethod(route.endpoint) or isinstance(route.endpoint, functools.partial):
        endpoint = route.endpoint.func if isinstance(route.endpoint, functools.partial) else route.endpoint
        path = self._remove_converter(route.path)
        for method in route.methods or ['GET']:
          if method == 'HEAD': continue
          endpoints_info.append(EndpointInfo(path, method.lower(), endpoint))
      else:
        path = self._remove_converter(route.path)
        for method in ['get', 'post', 'put', 'patch', 'delete', 'options']:
          if not hasattr(route.endpoint, method): continue
          func = getattr(route.endpoint, method)
          endpoints_info.append(EndpointInfo(path, method.lower(), func))
    return endpoints_info

  def get_schema(self, routes: list[Route], mount_path: str | None = None) -> dict[str, t.Any]:
    schema = dict(self.base_schema)
    schema.setdefault('paths', {})
    endpoints_info = self.get_endpoints(routes)
    if mount_path: mount_path = f'/{mount_path}' if not mount_path.startswith('/') else mount_path

    for endpoint in endpoints_info:
      parsed = self.parse_docstring(endpoint.func)
      if not parsed: continue

      path = endpoint.path if mount_path is None else mount_path + endpoint.path
      if path not in schema['paths']: schema['paths'][path] = {}
      schema['paths'][path][endpoint.http_method] = parsed

    return schema

def get_generator(title: str, components: list[AttrsInstance] | None = None, tags: list[dict[str, t.Any]] | None = None) -> SchemaGenerator:
  base_schema = dict(info={'title': title, 'version': API_VERSION}, version=OPENAPI_VERSION)
  if components: base_schema['components'] = {'schemas': {c.__name__: component_schema_generator(c) for c in components}}
  if tags: base_schema['tags'] = tags
  return OpenLLMSchemaGenerator(base_schema)

def component_schema_generator(attr_cls: type[AttrsInstance], description: str | None = None) -> dict[str, t.Any]:
  schema: dict[str, t.Any] = {'type': 'object', 'required': [], 'properties': {}, 'title': attr_cls.__name__}
  schema['description'] = first_not_none(getattr(attr_cls, '__doc__', None), description, default=f'Generated components for {attr_cls.__name__}')
  for field in attr.fields(attr.resolve_types(attr_cls)):
    attr_type = field.type
    origin_type = t.get_origin(attr_type)
    args_type = t.get_args(attr_type)

    # Map Python types to OpenAPI schema types
    if attr_type == str: schema_type = 'string'
    elif attr_type == int: schema_type = 'integer'
    elif attr_type == float: schema_type = 'number'
    elif attr_type == bool: schema_type = 'boolean'
    elif origin_type is list or origin_type is tuple:
      schema_type = 'array'
    elif origin_type is dict:
      schema_type = 'object'
      # Assuming string keys for simplicity, and handling Any type for values
      prop_schema = {
          'type': 'object',
          'additionalProperties':
              True if args_type[1] is t.Any else {
                  'type': 'string'
              }  # Simplified
      }
    elif attr_type == t.Optional[str]:
      schema_type = 'string'
    elif origin_type is t.Union and t.Any in args_type:
      schema_type = 'object'
      prop_schema = {
          'type': 'object',
          'additionalProperties': True  # Allows any type of values
      }
    else:
      schema_type = 'string'

    if 'prop_schema' not in locals(): prop_schema = {'type': schema_type}
    if field.default is not attr.NOTHING and not isinstance(field.default, attr.Factory): prop_schema['default'] = field.default
    if field.default is attr.NOTHING and not isinstance(attr_type, type(t.Optional)): schema['required'].append(field.name)
    schema['properties'][field.name] = prop_schema
    locals().pop('prop_schema', None)

  return schema

class MKSchema:
  def __init__(self, it: dict[str, t.Any]) -> None:
    self.it = it

  def asdict(self) -> dict[str, t.Any]:
    return self.it

def append_schemas(svc: bentoml.Service, generated_schema: dict[str, t.Any]) -> bentoml.Service:
  # HACK: Dirty hack to append schemas to existing service. We def need to support mounting Starlette app OpenAPI spec.
  from bentoml._internal.service.openapi.specification import OpenAPISpecification
  svc_schema = svc.openapi_spec
  if isinstance(svc_schema, (OpenAPISpecification, MKSchema)): svc_schema = svc_schema.asdict()
  if 'tags' in generated_schema: svc_schema['tags'].extend(generated_schema['tags'])
  if 'components' in generated_schema: svc_schema['components']['schemas'].update(generated_schema['components']['schemas'])
  svc_schema['paths'].update(generated_schema['paths'])
  # HACK: mk this attribute until we have a better way to add starlette schemas.
  from bentoml._internal.service import openapi
  openapi.generate_spec = lambda svc: MKSchema(svc_schema)
  OpenAPISpecification.asdict = lambda self: svc_schema
  return svc
