from __future__ import annotations
import functools
import logging
import time
import typing as t

from http import HTTPStatus

import orjson

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse
from starlette.routing import Route

from openllm.protocol.openai import ChatCompletionRequest
from openllm.protocol.openai import ChatCompletionResponse
from openllm.protocol.openai import ChatCompletionResponseChoice
from openllm.protocol.openai import ChatCompletionResponseStreamChoice
from openllm.protocol.openai import ChatCompletionStreamResponse
from openllm.protocol.openai import ChatMessage
from openllm.protocol.openai import CompletionRequest
from openllm.protocol.openai import Delta
from openllm.protocol.openai import ErrorResponse
from openllm.protocol.openai import ModelCard
from openllm.protocol.openai import ModelList
from openllm.protocol.openai import UsageInfo
from openllm.protocol.openai import get_conversation_prompt
from openllm_core import GenerationOutput
from openllm_core.utils import converter
from openllm_core.utils import gen_random_uuid

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
  from attr import AttrsInstance
  from starlette.requests import Request
  from starlette.responses import Response

  import bentoml
  import openllm

  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T

def jsonify_attr(obj: AttrsInstance) -> str:
  return orjson.dumps(converter.unstructure(obj)).decode()

def error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
  return JSONResponse({'error': converter.unstructure(ErrorResponse(message=message, type='invalid_request_error', code=status_code.value))}, status_code=status_code.value)

async def check_model(request: CompletionRequest | ChatCompletionRequest, model: str) -> JSONResponse | None:
  if request.model == model: return
  return error_response(
      HTTPStatus.NOT_FOUND,
      f"Model '{request.model}' does not exists. Try 'GET /v1/models' to see available models.\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request."
  )

def mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service:
  app = Starlette(debug=True,
                  routes=[
                      Route('/models', functools.partial(list_models, llm=llm), methods=['GET']),
                      Route('/completions', functools.partial(create_completions, llm=llm), methods=['POST']),
                      Route('/chat/completions', functools.partial(create_chat_completions, llm=llm), methods=['POST'])
                  ])
  svc.mount_asgi_app(app, path='/v1')
  return svc

# GET /v1/models
def list_models(_: Request, llm: openllm.LLM[M, T]) -> Response:
  return JSONResponse(converter.unstructure(ModelList(data=[ModelCard(id=llm.llm_type)])), status_code=HTTPStatus.OK.value)

# POST /v1/chat/completions
async def create_chat_completions(req: Request, llm: openllm.LLM[M, T]) -> Response:
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), ChatCompletionRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received.')
  logger.debug('Received chat completion request: %s', request)
  err_check = await check_model(request, llm.llm_type)
  if err_check is not None: return err_check

  prompt = await get_conversation_prompt(request, llm.config)
  model_name = request.model
  request_id = gen_random_uuid('chatcmpl')
  config = llm.config.with_openai_request(request)
  created_time = int(time.monotonic())

  result_generator = llm.generate_iterator(prompt, request_id=request_id, **config)

  def create_stream_response_json(index: int, text: str, finish_reason: str | None = None) -> str:
    choice_data = ChatCompletionResponseStreamChoice(index=index, delta=Delta(content=text), finish_reason=finish_reason)
    return jsonify_attr(ChatCompletionStreamResponse(id=request_id, created=created_time, model=model_name, choices=[choice_data]))

  async def completion_stream_generator() -> t.AsyncGenerator[str, None]:
    # first chunk with role
    for i in range(config['n']):
      choice_data = ChatCompletionResponseStreamChoice(index=i, delta=Delta(role='assistant'), finish_reason=None)
      chunk = ChatCompletionStreamResponse(id=request_id, choices=[choice_data], model=model_name)
      yield f'data: {jsonify_attr(chunk)}\n\n'

    previous_texts = [''] * config['n']
    previous_num_tokens = [0] * config['n']
    async for res in result_generator:
      for output in res.outputs:
        i = output.index
        delta_text = output.text[len(previous_texts[i]):]
        previous_texts[i] = output.text
        previous_num_tokens[i] += len(output.token_ids)
        yield f'data: {create_stream_response_json(i, delta_text)}\n\n'
        if output.finish_reason is not None:
          yield f'data: {create_stream_response_json(i, "", output.finish_reason)}\n\n'
    yield 'data: [DONE]\n\n'

  # Streaming case
  if request.stream: return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')

  # Non-streaming case
  final_result: GenerationOutput | None = None
  async for res in result_generator:
    if await req.is_disconnected(): return error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected.')
    final_result = t.cast(GenerationOutput, res)
  if final_result is None: return error_response(HTTPStatus.BAD_REQUEST, 'No response from model.')
  choices = [
      ChatCompletionResponseChoice(index=output.index, message=ChatMessage(role='assistant', content=output.text), finish_reason=output.finish_reason) for output in final_result.outputs
  ]
  num_prompt_tokens = len(final_result.prompt_token_ids)
  num_generated_tokens = sum(len(output.token_ids) for output in final_result.outputs)
  usage = UsageInfo(prompt_tokens=num_prompt_tokens, completion_tokens=num_generated_tokens, total_tokens=num_prompt_tokens + num_generated_tokens,)
  response = ChatCompletionResponse(id=request_id, created=created_time, model=model_name, usage=usage, choices=choices)

  if request.stream:
    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    async def fake_stream_generator() -> t.AsyncGenerator[str, None]:
      yield f'data: {jsonify_attr(response)}\n\n'
      yield 'data: [DONE]\n\n'

    return StreamingResponse(fake_stream_generator(), media_type='text/event-stream')

  return JSONResponse(jsonify_attr(response), status_code=HTTPStatus.OK.value)

# POST /v1/completions
async def create_completions(req: Request, llm: openllm.LLM[M, T]) -> Response:
  ...
