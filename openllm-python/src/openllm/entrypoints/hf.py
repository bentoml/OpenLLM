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

  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T

def mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service:
  app = Starlette(debug=True, routes=[Route('/agent', functools.partial(hf_agent, llm=llm), methods=['POST'])])
  svc.mount_asgi_app(app, path='/hf')
  return svc

async def hf_agent(request: Request, llm: openllm.LLM[M, T]) -> Response:
  json_str = await request.body()
  try:
    input_data = converter.structure(orjson.loads(json_str), openllm.HfAgentInput)
  except orjson.JSONDecodeError as err:
    raise openllm.exceptions.OpenLLMException(f'Invalid JSON input received: {err}') from None
  stop = input_data.parameters.pop('stop', ['\n'])
  try:
    return JSONResponse(await llm.runner.generate_one.async_run(input_data.inputs, stop, **input_data.parameters), status_code=200)
  except NotImplementedError:
    return JSONResponse(f"'{llm.llm_type}' is currently not supported with HuggingFace agents.", status_code=500)
