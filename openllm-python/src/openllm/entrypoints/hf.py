from __future__ import annotations
import functools
import typing as t

import orjson

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import openllm

from openllm_core.utils import converter

if t.TYPE_CHECKING:
  from starlette.requests import Request
  from starlette.responses import Response

  import bentoml

def mount_to_svc(svc: bentoml.Service, llm_runner: openllm.LLMRunner) -> bentoml.Service:
  app = Starlette(debug=True, routes=[Route('/agent', functools.partial(hf_agent, llm_runner=llm_runner), methods=['POST'])])
  svc.mount_asgi_app(app, path='/hf')
  return svc

async def hf_agent(request: Request, llm_runner: openllm.LLMRunner) -> Response:
  json_str = await request.body()
  try:
    input_data = converter.structure(orjson.loads(json_str), openllm.HfAgentInput)
  except orjson.JSONDecodeError as err:
    raise openllm.exceptions.OpenLLMException(f'Invalid JSON input received: {err}') from None
  stop = input_data.parameters.pop('stop', ['\n'])
  try:
    return JSONResponse(await llm_runner.generate_one.async_run(input_data.inputs, stop, **input_data.parameters), status_code=200)
  except NotImplementedError:
    return JSONResponse(f"'{llm_runner.llm_type}' is currently not supported with HuggingFace agents.", status_code=500)
