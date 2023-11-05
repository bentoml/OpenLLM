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

from openllm_core import GenerationOutput
from openllm_core.utils import converter
from openllm_core.utils import gen_random_uuid

from ._openapi import append_schemas
from ._openapi import get_generator
from ..protocol.openai import ChatCompletionRequest
from ..protocol.openai import ChatCompletionResponse
from ..protocol.openai import ChatCompletionResponseChoice
from ..protocol.openai import ChatCompletionResponseStreamChoice
from ..protocol.openai import ChatCompletionStreamResponse
from ..protocol.openai import ChatMessage
from ..protocol.openai import CompletionRequest
from ..protocol.openai import CompletionResponse
from ..protocol.openai import CompletionResponseChoice
from ..protocol.openai import CompletionResponseStreamChoice
from ..protocol.openai import CompletionStreamResponse
from ..protocol.openai import Delta
from ..protocol.openai import ErrorResponse
from ..protocol.openai import LogProbs
from ..protocol.openai import ModelCard
from ..protocol.openai import ModelList
from ..protocol.openai import UsageInfo
from ..protocol.openai import get_conversation_prompt

schemas = get_generator(
    'openai',
    components=[ErrorResponse, ModelList, ChatCompletionResponse, ChatCompletionRequest, ChatCompletionStreamResponse, CompletionRequest, CompletionResponse, CompletionStreamResponse],
    tags=[{
        'name': 'OpenAI',
        'description': 'OpenAI Compatible API support',
        'externalDocs': 'https://platform.openai.com/docs/api-reference/completions/object'
    }])
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

def create_logprobs(token_ids: list[int], id_logprobs: list[dict[int, float]], initial_text_offset: int = 0, *, llm: openllm.LLM[M, T]) -> LogProbs:
  # Create OpenAI-style logprobs.
  logprobs = LogProbs()
  last_token_len = 0
  for token_id, id_logprob in zip(token_ids, id_logprobs):
    token = llm.tokenizer.convert_ids_to_tokens(token_id)
    logprobs.tokens.append(token)
    logprobs.token_logprobs.append(id_logprob[token_id])
    if len(logprobs.text_offset) == 0:
      logprobs.text_offset.append(initial_text_offset)
    else:
      logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
    last_token_len = len(token)

    logprobs.top_logprobs.append({llm.tokenizer.convert_ids_to_tokens(i): p for i, p in id_logprob.items()})
  return logprobs

def mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service:
  app = Starlette(debug=True,
                  routes=[
                      Route('/models', functools.partial(list_models, llm=llm), methods=['GET']),
                      Route('/completions', functools.partial(create_completions, llm=llm), methods=['POST']),
                      Route('/chat/completions', functools.partial(create_chat_completions, llm=llm), methods=['POST'])
                  ])
  mount_path = '/v1'
  generated_schema = schemas.get_schema(routes=app.routes, mount_path=mount_path)
  svc.mount_asgi_app(app, path=mount_path)
  return append_schemas(svc, generated_schema)

# GET /v1/models
def list_models(_: Request, llm: openllm.LLM[M, T]) -> Response:
  """
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
    '200':
      description: The Model object
      content:
        application/json:
          example:
            id: davinci
            object: model
            created: 1686935002
            owned_by: openai
          schema:
            $ref: '#/components/schemas/ModelList'
  """
  return JSONResponse(converter.unstructure(ModelList(data=[ModelCard(id=llm.llm_type)])), status_code=HTTPStatus.OK.value)

# POST /v1/chat/completions
async def create_chat_completions(req: Request, llm: openllm.LLM[M, T]) -> Response:
  """
  ---
  consumes:
  - application/json
  description: >-
    Given a list of messages comprising a conversation, the model will return a
    response.
  operationId: openai__create_chat_completions
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
        schema:
          $ref: '#/components/schemas/ChatCompletionRequest'
  responses:
    '200':
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
            non-streaming:
              summary: Non-streaming output example
              value: >
                {"id": "chatcmpl-123", "object": "chat.completion", "created": 1677652288, "model": "gpt-3.5-turbo-0613", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there, how may I assist you today?"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}}
"""
  # TODO: Check for length based on model context_length
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), ChatCompletionRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received (Check server log).')
  logger.debug('Received chat completion request: %s', request)
  err_check = await check_model(request, llm.llm_type)
  if err_check is not None: return err_check

  model_name, request_id = request.model, gen_random_uuid('chatcmpl')
  created_time = int(time.monotonic())
  prompt = await get_conversation_prompt(request, llm.config)
  config = llm.config.with_openai_request(request)

  try:
    result_generator = llm.generate_iterator(prompt, request_id=request_id, format_prompt=False, **config)
  except Exception as err:
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, str(err))

  def create_stream_response_json(index: int, text: str, finish_reason: str | None = None) -> str:
    return jsonify_attr(
        ChatCompletionStreamResponse(id=request_id,
                                     created=created_time,
                                     model=model_name,
                                     choices=[ChatCompletionResponseStreamChoice(index=index, delta=Delta(content=text), finish_reason=finish_reason)]))

  async def completion_stream_generator() -> t.AsyncGenerator[str, None]:
    # first chunk with role
    for i in range(config['n']):
      yield f"data: {jsonify_attr(ChatCompletionStreamResponse(id=request_id, choices=[ChatCompletionResponseStreamChoice(index=i, delta=Delta(role='assistant'), finish_reason=None)], model=model_name))}\n\n"

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
  num_prompt_tokens, num_generated_tokens = len(final_result.prompt_token_ids), sum(len(output.token_ids) for output in final_result.outputs)
  usage = UsageInfo(prompt_tokens=num_prompt_tokens, completion_tokens=num_generated_tokens, total_tokens=num_prompt_tokens + num_generated_tokens)
  response = ChatCompletionResponse(id=request_id, created=created_time, model=model_name, usage=usage, choices=choices)

  if request.stream:
    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    async def fake_stream_generator() -> t.AsyncGenerator[str, None]:
      yield f'data: {jsonify_attr(response)}\n\n'
      yield 'data: [DONE]\n\n'

    return StreamingResponse(fake_stream_generator(), media_type='text/event-stream', status_code=HTTPStatus.OK.value)

  return JSONResponse(converter.unstructure(response), status_code=HTTPStatus.OK.value)

# POST /v1/completions
async def create_completions(req: Request, llm: openllm.LLM[M, T]) -> Response:
  """
  ---
  consumes:
    - application/json
  description: >-
    Given a prompt, the model will return one or more predicted completions, and
    can also return the probabilities of alternative tokens at each position. We
    recommend most users use our Chat completions API.
  operationId: openai__create_completions
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
  responses:
    '200':
      description: OK
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/CompletionResponse'
          examples:
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
            non-streaming:
              summary: Non-streaming output example
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
  """
  # TODO: Check for length based on model context_length
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), CompletionRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received (Check server log).')
  logger.debug('Received legacy completion request: %s', request)
  err_check = await check_model(request, llm.llm_type)
  if err_check is not None: return err_check

  if request.echo: return error_response(HTTPStatus.BAD_REQUEST, "'echo' is not yet supported.")
  if request.suffix is not None: return error_response(HTTPStatus.BAD_REQUEST, "'suffix' is not yet supported.")
  if request.logit_bias is not None and len(request.logit_bias) > 0: return error_response(HTTPStatus.BAD_REQUEST, "'logit_bias' is not yet supported.")

  if isinstance(request.prompt, list):
    if len(request.prompt) == 0: return error_response(HTTPStatus.BAD_REQUEST, 'Please provide at least one prompt..')
    if len(request.prompt) > 1: return error_response(HTTPStatus.BAD_REQUEST, 'Multiple prompts in a batch is not yet supported.')
    prompt = request.prompt[0]
  else:
    prompt = request.prompt

  model_name, request_id = request.model, gen_random_uuid('cmpl')
  created_time = int(time.monotonic())
  config = llm.config.with_openai_request(request)

  try:
    result_generator = llm.generate_iterator(prompt, request_id=request_id, **config)
  except Exception as err:
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, str(err))

  # best_of != n then we don't stream
  # TODO: support use_beam_search
  stream = request.stream and (config['best_of'] is None or config['n'] == config['best_of'])

  def create_stream_response_json(index: int, text: str, logprobs: LogProbs | None = None, finish_reason: t.Literal['stop', 'length'] | None = None) -> str:
    return jsonify_attr(
        CompletionStreamResponse(id=request_id,
                                 created=created_time,
                                 model=model_name,
                                 choices=[CompletionResponseStreamChoice(index=index, text=text, logprobs=logprobs, finish_reason=finish_reason)]))

  async def completion_stream_generator() -> t.AsyncGenerator[str, None]:
    previous_texts = [''] * config['n']
    previous_num_tokens = [0] * config['n']
    async for res in result_generator:
      for output in t.cast(GenerationOutput, res).outputs:
        i = output.index
        delta_text = output.text[len(previous_texts[i]):]
        if request.logprobs is not None:
          logprobs = create_logprobs(token_ids=output.token_ids[previous_num_tokens[i]:],
                                     id_logprobs=output.logprobs[previous_num_tokens[i]:],
                                     initial_text_offset=len(previous_texts[i]),
                                     llm=llm)
        else:
          logprobs = None
        previous_texts[i] = output.text
        previous_num_tokens[i] += len(output.token_ids)
        yield f'data: {create_stream_response_json(index=i, text=delta_text, logprobs=logprobs)}\n\n'
        if output.finish_reason is not None:
          logprobs = LogProbs() if request.logprobs is not None else None
          yield f'data: {create_stream_response_json(index=i, text="", logprobs=logprobs, finish_reason=output.finish_reason)}\n\n'
    yield 'data: [DONE]\n\n'

  # Streaming case
  if stream: return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')
  # Non-streaming case
  final_result: GenerationOutput | None = None
  async for res in result_generator:
    if await req.is_disconnected(): return error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected.')
    final_result = t.cast(GenerationOutput, res)
  if final_result is None: return error_response(HTTPStatus.BAD_REQUEST, 'No response from model.')

  choices: list[CompletionResponseChoice] = []
  for output in final_result.outputs:
    if request.logprobs is not None:
      logprobs = create_logprobs(token_ids=output.token_ids, id_logprobs=output.logprobs, llm=llm)
    else:
      logprobs = None
    choice_data = CompletionResponseChoice(index=output.index, text=output.text, logprobs=logprobs, finish_reason=output.finish_reason)
    choices.append(choice_data)

  num_prompt_tokens = len(final_result.prompt_token_ids)
  num_generated_tokens = sum(len(output.token_ids) for output in final_result.outputs)
  usage = UsageInfo(prompt_tokens=num_prompt_tokens, completion_tokens=num_generated_tokens, total_tokens=num_prompt_tokens + num_generated_tokens)
  response = CompletionResponse(id=request_id, created=created_time, model=model_name, usage=usage, choices=choices)

  if request.stream:
    # When user requests streaming but we don't stream, we still need to
    # return a streaming response with a single event.
    async def fake_stream_generator() -> t.AsyncGenerator[str, None]:
      yield f'data: {jsonify_attr(response)}\n\n'
      yield 'data: [DONE]\n\n'

    return StreamingResponse(fake_stream_generator(), media_type='text/event-stream', status_code=HTTPStatus.OK.value)

  return JSONResponse(converter.unstructure(response), status_code=HTTPStatus.OK.value)
