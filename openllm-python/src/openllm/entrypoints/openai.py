from __future__ import annotations
import logging
import time
import typing as t

from http import HTTPStatus

import orjson

from starlette.responses import JSONResponse

from openllm.protocol.openai import ChatCompletionRequest
from openllm.protocol.openai import ChatCompletionResponseStreamChoice
from openllm.protocol.openai import ChatCompletionStreamResponse
from openllm.protocol.openai import CompletionRequest
from openllm.protocol.openai import Delta
from openllm.protocol.openai import ErrorResponse
from openllm.protocol.openai import ModelCard
from openllm.protocol.openai import ModelList
from openllm.protocol.openai import get_conversation_prompt
from openllm_core.utils import bentoml_cattr as cattr
from openllm_core.utils import first_not_none
from openllm_core.utils import gen_random_uuid

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING: from starlette.requests import Request; import openllm, bentoml

def error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
  return JSONResponse(cattr.unstructure(ErrorResponse(message=message, type='invalid_request_error')), status_code=status_code.value)

async def check_model(request: CompletionRequest | ChatCompletionRequest, model: str) -> JSONResponse | None:
  if request.model == model: return
  return error_response(
      HTTPStatus.NOT_FOUND,
      "Model '%s' does not exists. Try 'GET /v1/models' to see available models.\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request.")

def mount_to_svc(svc: bentoml.Service, llm_runner: openllm.LLMRunner) -> None:

  # GET /v1/models
  def list_models(_: Request) -> Response:
    return JSONResponse(cattr.unstructure(ModelList(data=[ModelCard(id=runner.llm_type)])), status_code=HTTPStatus.OK.value)

# POST /v1/chat/completions
async def create_chat_completions(req: Request, llm_runner: openllm.LLMRunner):
  json_str = await req.body()
  try:
    request = cattr.structure(orjson.loads(json_str), ChatCompletionRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received.')
  logger.debug('Received chat completion request: %s', request)
  err_check = await check_model(request, llm_runner.llm_type)
  if err_check is not None: return err_check

  await get_conversation_prompt(request)
  model_name = request.model
  request_id = gen_random_uuid('chatcmpl')
  created_time = int(time.monotonic())
  first_not_none(request.temperature, default=llm_runner.config['temperature'])
  first_not_none(request.top_p, default=llm_runner.config['top_p'])
  first_not_none(request.n, default=llm_runner.config['n'])
  first_not_none(request.stop, default=llm_runner.config['stop'])
  first_not_none(request.max_tokens, default=llm_runner.config['max_new_tokens'])
  first_not_none(request.presence_penalty, default=llm_runner.config['presence_penalty'])
  first_not_none(request.frequency_penalty, default=llm_runner.config['frequency_penalty'])

  def create_stream_response_json(index: int, text: str, finish_reason: str | None = None) -> str:
    choice_data = ChatCompletionResponseStreamChoice(index=index, delta=Delta(content=text), finish_reason=finish_reason)
    response = ChatCompletionStreamResponse(id=request_id, created=created_time, model=model_name, choices=[choice_data])
    return orjson.dumps(cattr.unstructure(response)).decode()

  def completion_stream_generator() -> t.AsyncGenerator[str, None]:
    # first chunk with role
    for i in range(request.n):
      choice_data = ChatCompletionResponseStreamChoice(index=i, delta=Delta(role='assistant'), finish_reason=None)
      ChatCompletionStreamResponse(id=request_id, choices=[choice_data], model=model_name)
