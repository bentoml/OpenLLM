from __future__ import annotations
import functools
import json
import logging
import traceback
import typing as t
from http import HTTPStatus

import orjson
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from openllm_core.utils import converter, gen_random_uuid

from ._openapi import add_schema_definitions, append_schemas, get_generator
from ..protocol.cohere import (
  Chat,
  ChatStreamEnd,
  ChatStreamStart,
  ChatStreamTextGeneration,
  CohereChatRequest,
  CohereErrorResponse,
  CohereGenerateRequest,
  Generation,
  Generations,
  StreamingGenerations,
  StreamingText,
)

schemas = get_generator(
  'cohere',
  components=[
    CohereChatRequest,
    CohereErrorResponse,
    CohereGenerateRequest,
    Generation,
    Generations,
    StreamingGenerations,
    StreamingText,
    Chat,
    ChatStreamStart,
    ChatStreamEnd,
    ChatStreamTextGeneration,
  ],
  tags=[
    {
      'name': 'Cohere',
      'description': 'Cohere compatible API. Currently support /generate, /chat',
      'externalDocs': 'https://docs.cohere.com/docs/the-cohere-platform',
    }
  ],
  inject=False,
)
logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
  from attr import AttrsInstance
  from starlette.requests import Request
  from starlette.responses import Response

  import bentoml
  import openllm
  from openllm_core._schemas import GenerationOutput
  from openllm_core._typing_compat import M, T


def jsonify_attr(obj: AttrsInstance) -> str:
  return json.dumps(converter.unstructure(obj))


def error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
  return JSONResponse(converter.unstructure(CohereErrorResponse(text=message)), status_code=status_code.value)


async def check_model(request: CohereGenerateRequest | CohereChatRequest, model: str) -> JSONResponse | None:
  if request.model is None or request.model == model:
    return None
  return error_response(
    HTTPStatus.NOT_FOUND,
    f"Model '{request.model}' does not exists. Try 'GET /v1/models' to see current running models.",
  )


def mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service:
  app = Starlette(
    debug=True,
    routes=[
      Route(
        '/v1/generate', endpoint=functools.partial(cohere_generate, llm=llm), name='cohere_generate', methods=['POST']
      ),
      Route('/v1/chat', endpoint=functools.partial(cohere_chat, llm=llm), name='cohere_chat', methods=['POST']),
      Route('/schema', endpoint=openapi_schema, include_in_schema=False),
    ],
  )
  mount_path = '/cohere'

  svc.mount_asgi_app(app, path=mount_path)
  return append_schemas(
    svc, schemas.get_schema(routes=app.routes, mount_path=mount_path), tags_order='append', inject=False
  )


@add_schema_definitions
async def cohere_generate(req: Request, llm: openllm.LLM[M, T]) -> Response:
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), CohereGenerateRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received (Check server log).')
  logger.debug('Received generate request: %s', request)

  err_check = await check_model(request, llm.llm_type)
  if err_check is not None:
    return err_check

  request_id = gen_random_uuid('cohere-generate')
  config = llm.config.with_request(request)

  if request.prompt_vars is not None:
    prompt = request.prompt.format(**request.prompt_vars)
  else:
    prompt = request.prompt

  # TODO: support end_sequences, stop_sequences, logit_bias, return_likelihoods, truncate

  try:
    result_generator = llm.generate_iterator(prompt, request_id=request_id, stop=request.stop_sequences, **config)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')

  def create_stream_response_json(index: int, text: str, is_finished: bool) -> str:
    return f'{jsonify_attr(StreamingText(index=index, text=text, is_finished=is_finished))}\n'

  async def generate_stream_generator() -> t.AsyncGenerator[str, None]:
    async for res in result_generator:
      for output in res.outputs:
        yield create_stream_response_json(index=output.index, text=output.text, is_finished=output.finish_reason)

  try:
    # streaming case
    if request.stream:
      return StreamingResponse(generate_stream_generator(), media_type='text/event-stream')
    # None-streaming case
    final_result: GenerationOutput | None = None
    texts: list[list[str]] = [[]] * config['num_generations']
    token_ids: list[list[int]] = [[]] * config['num_generations']
    async for res in result_generator:
      if await req.is_disconnected():
        return error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected.')
      for output in res.outputs:
        texts[output.index].append(output.text)
        token_ids[output.index].extend(output.token_ids)
      final_result = res
    if final_result is None:
      return error_response(HTTPStatus.BAD_REQUEST, 'No response from model.')
    final_result = final_result.with_options(
      outputs=[
        output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index])
        for output in final_result.outputs
      ]
    )
    response = Generations(
      id=request_id,
      generations=[
        Generation(id=request_id, text=output.text, prompt=prompt, finish_reason=output.finish_reason)
        for output in final_result.outputs
      ],
    )
    return JSONResponse(converter.unstructure(response), status_code=HTTPStatus.OK.value)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')


def _transpile_cohere_chat_messages(request: CohereChatRequest) -> list[dict[str, str]]:
  def convert_role(role):
    return {'User': 'user', 'Chatbot': 'assistant'}[role]

  chat_history = request.chat_history
  if chat_history:
    messages = [{'role': convert_role(msg['role']), 'content': msg['message']} for msg in chat_history]
  else:
    messages = []
  messages.append({'role': 'user', 'content': request.message})
  return messages


@add_schema_definitions
async def cohere_chat(req: Request, llm: openllm.LLM[M, T]) -> Response:
  json_str = await req.body()
  try:
    request = converter.structure(orjson.loads(json_str), CohereChatRequest)
  except orjson.JSONDecodeError as err:
    logger.debug('Sent body: %s', json_str)
    logger.error('Invalid JSON input received: %s', err)
    return error_response(HTTPStatus.BAD_REQUEST, 'Invalid JSON input received (Check server log).')
  logger.debug('Received chat completion request: %s', request)

  err_check = await check_model(request, llm.llm_type)
  if err_check is not None:
    return err_check

  request_id = gen_random_uuid('cohere-chat')
  prompt: str = llm.tokenizer.apply_chat_template(
    _transpile_cohere_chat_messages(request), tokenize=False, add_generation_prompt=llm.config['add_generation_prompt']
  )
  logger.debug('Prompt: %r', prompt)
  config = llm.config.with_request(request)

  try:
    result_generator = llm.generate_iterator(prompt, request_id=request_id, **config)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')

  def create_stream_generation_json(index: int, text: str, is_finished: bool) -> str:
    return f'{jsonify_attr(ChatStreamTextGeneration(index=index, text=text, is_finished=is_finished))}\n'

  async def completion_stream_generator() -> t.AsyncGenerator[str, None]:
    texts: list[str] = []
    token_ids: list[int] = []
    yield f'{jsonify_attr(ChatStreamStart(is_finished=False, index=0, generation_id=request_id))}\n'

    it = None
    async for res in result_generator:
      yield create_stream_generation_json(index=res.outputs[0].index, text=res.outputs[0].text, is_finished=False)
      texts.append(res.outputs[0].text)
      token_ids.extend(res.outputs[0].token_ids)
      it = res

    if it is None:
      raise ValueError('No response from model.')
    num_prompt_tokens = len(t.cast(t.List[int], it.prompt_token_ids))
    num_response_tokens = len(token_ids)
    total_tokens = num_prompt_tokens + num_response_tokens

    json_str = jsonify_attr(
      ChatStreamEnd(
        is_finished=True,
        finish_reason='COMPLETE',
        index=0,
        response=Chat(
          response_id=request_id,
          message=request.message,
          text=''.join(texts),
          prompt=prompt,
          chat_history=request.chat_history,
          token_count={
            'prompt_tokens': num_prompt_tokens,
            'response_tokens': num_response_tokens,
            'total_tokens': total_tokens,
          },
        ),
      )
    )
    yield f'{json_str}\n'

  try:
    if request.stream:
      return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')
    # Non-streaming case
    final_result: GenerationOutput | None = None
    texts: list[str] = []
    token_ids: list[int] = []
    async for res in result_generator:
      if await req.is_disconnected():
        return error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected.')
      texts.append(res.outputs[0].text)
      token_ids.extend(res.outputs[0].token_ids)
      final_result = res
    if final_result is None:
      return error_response(HTTPStatus.BAD_REQUEST, 'No response from model.')
    final_result = final_result.with_options(
      outputs=[final_result.outputs[0].with_options(text=''.join(texts), token_ids=token_ids)]
    )
    num_prompt_tokens = len(t.cast(t.List[int], final_result.prompt_token_ids))
    num_response_tokens = len(token_ids)
    total_tokens = num_prompt_tokens + num_response_tokens

    response = Chat(
      response_id=request_id,
      message=request.message,
      text=''.join(texts),
      prompt=prompt,
      chat_history=request.chat_history,
      token_count={
        'prompt_tokens': num_prompt_tokens,
        'response_tokens': num_response_tokens,
        'total_tokens': total_tokens,
      },
    )
    return JSONResponse(converter.unstructure(response), status_code=HTTPStatus.OK.value)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')


def openapi_schema(req: Request) -> Response:
  return schemas.OpenAPIResponse(req)
