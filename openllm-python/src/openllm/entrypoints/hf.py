from __future__ import annotations
import functools
import logging
import typing as t

from http import HTTPStatus

import orjson

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from openllm_core.utils import converter

from ._openapi import HF_AGENT_SCHEMA
from ._openapi import add_schema_definitions
from ._openapi import append_schemas
from ._openapi import get_generator
from ..protocol.hf import AgentErrorResponse
from ..protocol.hf import AgentRequest
from ..protocol.hf import AgentResponse

schemas = get_generator('hf',
                        components=[AgentRequest, AgentResponse, AgentErrorResponse],
                        tags=[{
                            'name': 'HF',
                            'description': 'Includes HF Agent support',
                            'externalDocs': 'https://huggingface.co/docs/transformers/main_classes/agent'
                        }])
logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
  from starlette.requests import Request
  from starlette.responses import Response

  import bentoml
  import openllm

  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T

def mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service:
  app = Starlette(
      debug=True,
      routes=[Route('/agent', endpoint=functools.partial(hf_agent, llm=llm), name='hf_agent', methods=['POST']),
              Route('/schema', endpoint=openapi_schema, include_in_schema=False)])
  mount_path = '/hf'
  generated_schema = schemas.get_schema(routes=app.routes, mount_path=mount_path)
  svc.mount_asgi_app(app, path=mount_path)
  return append_schemas(svc, generated_schema, tags_order='append')

def error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
  return JSONResponse(converter.unstructure(AgentErrorResponse(message=message, error_code=status_code.value)), status_code=status_code.value)

@add_schema_definitions(HF_AGENT_SCHEMA)
async def hf_agent(req: Request, llm: openllm.LLM[M, T]) -> Response:
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), AgentRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received (Check server log).')

  stop = request.parameters.pop('stop', ['\n'])
  try:
    result: str = await llm.generate(request.inputs, stop=stop, **request.parameters)
    return JSONResponse(converter.unstructure([AgentResponse(generated_text=result.outputs[0].text)]), status_code=HTTPStatus.OK.value)
  except Exception as err:
    logger.error('Error while generating: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'Error while generating (Check server log).')

def openapi_schema(req: Request) -> Response:
  return schemas.OpenAPIResponse(req)
