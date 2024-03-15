import functools, logging
from http import HTTPStatus
import orjson
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from openllm_core.utils import converter
from ._openapi import add_schema_definitions, append_schemas, get_generator
from ..protocol.hf import AgentRequest, AgentResponse, HFErrorResponse

schemas = get_generator(
  'hf',
  components=[AgentRequest, AgentResponse, HFErrorResponse],
  tags=[
    {
      'name': 'HF',
      'description': 'HF integration, including Agent and others schema endpoints.',
      'externalDocs': 'https://huggingface.co/docs/transformers/main_classes/agent',
    }
  ],
)
logger = logging.getLogger(__name__)


def mount_to_svc(svc, llm):
  app = Starlette(
    debug=True,
    routes=[
      Route('/agent', endpoint=functools.partial(hf_agent, llm=llm), name='hf_agent', methods=['POST']),
      Route('/adapters', endpoint=functools.partial(hf_adapters, llm=llm), name='adapters', methods=['GET']),
      Route('/schema', endpoint=lambda req: schemas.OpenAPIResponse(req), include_in_schema=False),
    ],
  )
  mount_path = '/hf'
  svc.mount_asgi_app(app, path=mount_path)
  return append_schemas(svc, schemas.get_schema(routes=app.routes, mount_path=mount_path), tags_order='append')


def error_response(status_code, message):
  return JSONResponse(converter.unstructure(HFErrorResponse(message=message, error_code=status_code.value)), status_code=status_code.value)


@add_schema_definitions
async def hf_agent(req, llm):
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), AgentRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received (Check server log).')

  stop = request.parameters.pop('stop', [])
  try:
    result = await llm.generate(request.inputs, stop=stop, **request.parameters)
    return JSONResponse(converter.unstructure([AgentResponse(generated_text=result.outputs[0].text)]), status_code=HTTPStatus.OK.value)
  except Exception as err:
    logger.error('Error while generating: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'Error while generating (Check server log).')


@add_schema_definitions
def hf_adapters(req, llm):
  if not llm.has_adapters:
    return error_response(HTTPStatus.NOT_FOUND, 'No adapters found.')
  return JSONResponse(
    {
      adapter_tuple[1]: {'adapter_name': k, 'adapter_type': adapter_tuple[0].peft_type.value}
      for k, adapter_tuple in dict(*llm.adapter_map.values()).items()
    },
    status_code=HTTPStatus.OK.value,
  )
