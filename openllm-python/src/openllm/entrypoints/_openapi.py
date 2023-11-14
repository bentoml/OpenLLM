from __future__ import annotations
import functools
import inspect
import typing as t

import attr
from starlette.routing import BaseRoute, Host, Mount, Route
from starlette.schemas import EndpointInfo, SchemaGenerator

from openllm_core._typing_compat import ParamSpec
from openllm_core.utils import first_not_none

if t.TYPE_CHECKING:
  from attr import AttrsInstance

  import bentoml

P = ParamSpec('P')
OPENAPI_VERSION, API_VERSION = '3.0.2', '1.0'
# NOTE: OpenAI schema
LIST_MODELS_SCHEMA = """\
---
consumes:
- application/json
description: >
  List and describe the various models available in the API.

  You can refer to the available supported models with `openllm models` for more
  information.
operationId: openai__list_models
produces:
  - application/json
summary: Describes a model offering that can be used with the API.
tags:
  - OpenAI
x-bentoml-name: list_models
responses:
  200:
    description: The Model object
    content:
      application/json:
        example:
          object: 'list'
          data:
            - id: meta-llama--Llama-2-13b-chat-hf
              object: model
              created: 1686935002
              owned_by: 'na'
        schema:
          $ref: '#/components/schemas/ModelList'
"""
CHAT_COMPLETIONS_SCHEMA = """\
---
consumes:
- application/json
description: >-
  Given a list of messages comprising a conversation, the model will return a
  response.
operationId: openai__chat_completions
produces:
  - application/json
tags:
  - OpenAI
x-bentoml-name: create_chat_completions
summary: Creates a model response for the given chat conversation.
requestBody:
  required: true
  content:
    application/json:
      examples:
        one-shot:
          summary: One-shot input example
          value:
            messages:
              - role: system
                content: You are a helpful assistant.
              - role: user
                content: Hello, I'm looking for a chatbot that can help me with my work.
            model: meta-llama--Llama-2-13b-chat-hf
            max_tokens: 256
            temperature: 0.7
            top_p: 0.43
            n: 1
            stream: false
        streaming:
          summary: Streaming input example
          value:
            messages:
              - role: system
                content: You are a helpful assistant.
              - role: user
                content: Hello, I'm looking for a chatbot that can help me with my work.
            model: meta-llama--Llama-2-13b-chat-hf
            max_tokens: 256
            temperature: 0.7
            top_p: 0.43
            n: 1
            stream: true
            stop:
              - "\\n"
              - "<|endoftext|>"
      schema:
        $ref: '#/components/schemas/ChatCompletionRequest'
responses:
  200:
    description: OK
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ChatCompletionResponse'
        examples:
          streaming:
            summary: Streaming output example
            value: >
              {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
          one-shot:
            summary: One-shot output example
            value: >
              {"id": "chatcmpl-123", "object": "chat.completion", "created": 1677652288, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there, how may I assist you today?"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}}
  404:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ErrorResponse'
        examples:
          wrong-model:
            summary: Wrong model
            value: >
              {
                "error": {
                  "message": "Model 'meta-llama--Llama-2-13b-chat-hf' does not exists. Try 'GET /v1/models' to see available models.\\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request.",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 404
                }
              }
    description: NotFound
  500:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ErrorResponse'
        examples:
          invalid-parameters:
            summary: Invalid parameters
            value: >
              {
                "error": {
                  "message": "`top_p` has to be a float > 0 and < 1, but is 4.0",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 500
                }
              }
    description: Internal Server Error
  400:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ErrorResponse'
        examples:
          invalid-json:
            summary: Invalid JSON sent
            value: >
              {
                "error": {
                  "message": "Invalid JSON input received (Check server log).",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 400
                }
              }
          invalid-prompt:
            summary: Invalid prompt
            value: >
              {
                "error": {
                  "message": "Please provide a prompt.",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 400
                }
              }
    description: Bad Request
"""
COMPLETIONS_SCHEMA = """\
---
consumes:
  - application/json
description: >-
  Given a prompt, the model will return one or more predicted completions, and
  can also return the probabilities of alternative tokens at each position. We
  recommend most users use our Chat completions API.
operationId: openai__completions
produces:
  - application/json
tags:
  - OpenAI
x-bentoml-name: create_completions
summary: Creates a completion for the provided prompt and parameters.
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/CompletionRequest'
      examples:
        one-shot:
          summary: One-shot input example
          value:
            prompt: This is a test
            model: meta-llama--Llama-2-13b-chat-hf
            max_tokens: 256
            temperature: 0.7
            logprobs: 1
            top_p: 0.43
            n: 1
            stream: false
        streaming:
          summary: Streaming input example
          value:
            prompt: This is a test
            model: meta-llama--Llama-2-13b-chat-hf
            max_tokens: 256
            temperature: 0.7
            top_p: 0.43
            logprobs: 1
            n: 1
            stream: true
            stop:
              - "\\n"
              - "<|endoftext|>"
responses:
  200:
    description: OK
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/CompletionResponse'
        examples:
          one-shot:
            summary: One-shot output example
            value:
              id: cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7
              object: text_completion
              created: 1589478378
              model: VAR_model_id
              choices:
                - text: This is indeed a test
                  index: 0
                  logprobs: null
                  finish_reason: length
              usage:
                prompt_tokens: 5
                completion_tokens: 7
                total_tokens: 12
          streaming:
            summary: Streaming output example
            value:
              id: cmpl-7iA7iJjj8V2zOkCGvWF2hAkDWBQZe
              object: text_completion
              created: 1690759702
              choices:
                - text: This
                  index: 0
                  logprobs: null
                  finish_reason: null
              model: gpt-3.5-turbo-instruct
  404:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ErrorResponse'
        examples:
          wrong-model:
            summary: Wrong model
            value: >
              {
                "error": {
                  "message": "Model 'meta-llama--Llama-2-13b-chat-hf' does not exists. Try 'GET /v1/models' to see available models.\\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request.",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 404
                }
              }
    description: NotFound
  500:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ErrorResponse'
        examples:
          invalid-parameters:
            summary: Invalid parameters
            value: >
              {
                "error": {
                  "message": "`top_p` has to be a float > 0 and < 1, but is 4.0",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 500
                }
              }
    description: Internal Server Error
  400:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ErrorResponse'
        examples:
          invalid-json:
            summary: Invalid JSON sent
            value: >
              {
                "error": {
                  "message": "Invalid JSON input received (Check server log).",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 400
                }
              }
          invalid-prompt:
            summary: Invalid prompt
            value: >
              {
                "error": {
                  "message": "Please provide a prompt.",
                  "type": "invalid_request_error",
                  "object": "error",
                  "param": null,
                  "code": 400
                }
              }
    description: Bad Request
"""
HF_AGENT_SCHEMA = """\
---
consumes:
  - application/json
description: Generate instruction for given HF Agent chain for all OpenLLM supported models.
operationId: hf__agent
summary: Generate instruction for given HF Agent.
tags:
  - HF
x-bentoml-name: hf_agent
produces:
  - application/json
requestBody:
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/AgentRequest'
      example:
        inputs: "Is the following `text` positive or negative?"
        parameters:
          text: "This is a positive text."
          stop: ["\n"]
  required: true
responses:
  200:
    description: Successfull generated instruction.
    content:
      application/json:
        example:
          - generated_text: "This is a generated instruction."
        schema:
          $ref: '#/components/schemas/AgentResponse'
  400:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/HFErrorResponse'
    description: Bad Request
  500:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/HFErrorResponse'
    description: Not Found
"""
HF_ADAPTERS_SCHEMA = """\
---
consumes:
- application/json
description: Return current list of adapters for given LLM.
operationId: hf__adapters_map
produces:
  - application/json
summary: Describes a model offering that can be used with the API.
tags:
  - HF
x-bentoml-name: hf_adapters
responses:
  200:
    description: Return list of LoRA adapters.
    content:
      application/json:
        example:
          aarnphm/opt-6-7b-quotes:
            adapter_name: default
            adapter_type: LORA
          aarnphm/opt-6-7b-dolly:
            adapter_name: dolly
            adapter_type: LORA
  500:
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/HFErrorResponse'
    description: Not Found
"""
COHERE_GENERATE_SCHEMA = """\
---
consumes:
  - application/json
description: >-
  Given a prompt, the model will return one or more predicted completions, and
  can also return the probabilities of alternative tokens at each position.
operationId: cohere__generate
produces:
  - application/json
tags:
  - Cohere
x-bentoml-name: cohere_generate
summary: Creates a completion for the provided prompt and parameters.
requestBody:
  required: true
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/CohereGenerateRequest'
      examples:
        one-shot:
          summary: One-shot input example
          value:
            prompt: This is a test
            max_tokens: 256
            temperature: 0.7
            p: 0.43
            k: 12
            num_generations: 2
            stream: false
        streaming:
          summary: Streaming input example
          value:
            prompt: This is a test
            max_tokens: 256
            temperature: 0.7
            p: 0.43
            k: 12
            num_generations: 2
            stream: true
            stop_sequences:
              - "\\n"
              - "<|endoftext|>"
"""
COHERE_CHAT_SCHEMA = """\
---
consumes:
- application/json
description: >-
  Given a list of messages comprising a conversation, the model will return a response.
operationId: cohere__chat
produces:
  - application/json
tags:
  - Cohere
x-bentoml-name: cohere_chat
summary: Creates a model response for the given chat conversation.
"""

_SCHEMAS = {k[:-7].lower(): v for k, v in locals().items() if k.endswith('_SCHEMA')}


def add_schema_definitions(func: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
  append_str = _SCHEMAS.get(func.__name__.lower(), '')
  if not append_str:
    return func
  if func.__doc__ is None:
    func.__doc__ = ''
  func.__doc__ = func.__doc__.strip() + '\n\n' + append_str.strip()
  return func


class OpenLLMSchemaGenerator(SchemaGenerator):
  def get_endpoints(self, routes: list[BaseRoute]) -> list[EndpointInfo]:
    endpoints_info: list[EndpointInfo] = []
    for route in routes:
      if isinstance(route, (Mount, Host)):
        routes = route.routes or []
        path = self._remove_converter(route.path) if isinstance(route, Mount) else ''
        sub_endpoints = [
          EndpointInfo(path=f'{path}{sub_endpoint.path}', http_method=sub_endpoint.http_method, func=sub_endpoint.func)
          for sub_endpoint in self.get_endpoints(routes)
        ]
        endpoints_info.extend(sub_endpoints)
      elif not isinstance(route, Route) or not route.include_in_schema:
        continue
      elif (
        inspect.isfunction(route.endpoint)
        or inspect.ismethod(route.endpoint)
        or isinstance(route.endpoint, functools.partial)
      ):
        endpoint = route.endpoint.func if isinstance(route.endpoint, functools.partial) else route.endpoint
        path = self._remove_converter(route.path)
        for method in route.methods or ['GET']:
          if method == 'HEAD':
            continue
          endpoints_info.append(EndpointInfo(path, method.lower(), endpoint))
      else:
        path = self._remove_converter(route.path)
        for method in ['get', 'post', 'put', 'patch', 'delete', 'options']:
          if not hasattr(route.endpoint, method):
            continue
          func = getattr(route.endpoint, method)
          endpoints_info.append(EndpointInfo(path, method.lower(), func))
    return endpoints_info

  def get_schema(self, routes: list[BaseRoute], mount_path: str | None = None) -> dict[str, t.Any]:
    schema = dict(self.base_schema)
    schema.setdefault('paths', {})
    endpoints_info = self.get_endpoints(routes)
    if mount_path:
      mount_path = f'/{mount_path}' if not mount_path.startswith('/') else mount_path

    for endpoint in endpoints_info:
      parsed = self.parse_docstring(endpoint.func)
      if not parsed:
        continue

      path = endpoint.path if mount_path is None else mount_path + endpoint.path
      if path not in schema['paths']:
        schema['paths'][path] = {}
      schema['paths'][path][endpoint.http_method] = parsed

    return schema


def get_generator(
  title: str,
  components: list[type[AttrsInstance]] | None = None,
  tags: list[dict[str, t.Any]] | None = None,
  inject: bool = True,
) -> OpenLLMSchemaGenerator:
  base_schema: dict[str, t.Any] = dict(info={'title': title, 'version': API_VERSION}, version=OPENAPI_VERSION)
  if components and inject:
    base_schema['components'] = {'schemas': {c.__name__: component_schema_generator(c) for c in components}}
  if tags is not None and tags and inject:
    base_schema['tags'] = tags
  return OpenLLMSchemaGenerator(base_schema)


def component_schema_generator(attr_cls: type[AttrsInstance], description: str | None = None) -> dict[str, t.Any]:
  schema: dict[str, t.Any] = {'type': 'object', 'required': [], 'properties': {}, 'title': attr_cls.__name__}
  schema['description'] = first_not_none(
    getattr(attr_cls, '__doc__', None), description, default=f'Generated components for {attr_cls.__name__}'
  )
  for field in attr.fields(attr.resolve_types(attr_cls)):  # type: ignore[misc,type-var]
    attr_type = field.type
    origin_type = t.get_origin(attr_type)
    args_type = t.get_args(attr_type)

    # Map Python types to OpenAPI schema types
    if attr_type == str:
      schema_type = 'string'
    elif attr_type == int:
      schema_type = 'integer'
    elif attr_type == float:
      schema_type = 'number'
    elif attr_type == bool:
      schema_type = 'boolean'
    elif origin_type is list or origin_type is tuple:
      schema_type = 'array'
    elif origin_type is dict:
      schema_type = 'object'
      # Assuming string keys for simplicity, and handling Any type for values
      prop_schema = {'type': 'object', 'additionalProperties': True if args_type[1] is t.Any else {'type': 'string'}}
    elif attr_type == t.Optional[str]:
      schema_type = 'string'
    elif origin_type is t.Union and t.Any in args_type:
      schema_type = 'object'
      prop_schema = {'type': 'object', 'additionalProperties': True}
    else:
      schema_type = 'string'

    if 'prop_schema' not in locals():
      prop_schema = {'type': schema_type}
    if field.default is not attr.NOTHING and not isinstance(field.default, attr.Factory):
      prop_schema['default'] = field.default  # type: ignore[arg-type]
    if field.default is attr.NOTHING and not isinstance(attr_type, type(t.Optional)):
      schema['required'].append(field.name)
    schema['properties'][field.name] = prop_schema
    locals().pop('prop_schema', None)

  return schema


class MKSchema:
  def __init__(self, it: dict[str, t.Any]) -> None:
    self.it = it

  def asdict(self) -> dict[str, t.Any]:
    return self.it


def append_schemas(
  svc: bentoml.Service,
  generated_schema: dict[str, t.Any],
  tags_order: t.Literal['prepend', 'append'] = 'prepend',
  inject: bool = True,
) -> bentoml.Service:
  # HACK: Dirty hack to append schemas to existing service. We def need to support mounting Starlette app OpenAPI spec.
  from bentoml._internal.service.openapi.specification import OpenAPISpecification

  if not inject:
    return svc

  svc_schema = svc.openapi_spec
  if isinstance(svc_schema, (OpenAPISpecification, MKSchema)):
    svc_schema = svc_schema.asdict()
  if 'tags' in generated_schema:
    if tags_order == 'prepend':
      svc_schema['tags'] = generated_schema['tags'] + svc_schema['tags']
    elif tags_order == 'append':
      svc_schema['tags'].extend(generated_schema['tags'])
    else:
      raise ValueError(f'Invalid tags_order: {tags_order}')
  if 'components' in generated_schema:
    svc_schema['components']['schemas'].update(generated_schema['components']['schemas'])
  svc_schema['paths'].update(generated_schema['paths'])

  # HACK: mk this attribute until we have a better way to add starlette schemas.
  from bentoml._internal.service import openapi

  def mk_generate_spec(svc, openapi_version=OPENAPI_VERSION):
    return MKSchema(svc_schema)

  def mk_asdict(self):
    return svc_schema

  openapi.generate_spec = mk_generate_spec
  OpenAPISpecification.asdict = mk_asdict
  return svc
