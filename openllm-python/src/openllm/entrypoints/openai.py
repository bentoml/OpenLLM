import functools
import logging
import time
import traceback
from http import HTTPStatus

import orjson
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from openllm_core.utils import converter, gen_random_uuid

from ._openapi import add_schema_definitions, append_schemas, apply_schema, get_generator
from ..protocol.openai import (
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionResponseChoice,
  ChatCompletionResponseStreamChoice,
  ChatCompletionStreamResponse,
  ChatMessage,
  CompletionRequest,
  CompletionResponse,
  CompletionResponseChoice,
  CompletionResponseStreamChoice,
  CompletionStreamResponse,
  Delta,
  ErrorResponse,
  LogProbs,
  ModelCard,
  ModelList,
  UsageInfo,
)

schemas = get_generator(
  'openai',
  components=[
    ErrorResponse,
    ModelList,
    ChatCompletionResponse,
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
  ],
  tags=[
    {
      'name': 'OpenAI',
      'description': 'OpenAI Compatible API support',
      'externalDocs': 'https://platform.openai.com/docs/api-reference/completions/object',
    }
  ],
)
logger = logging.getLogger(__name__)


def jsonify_attr(obj):
  return orjson.dumps(converter.unstructure(obj)).decode()


def error_response(status_code, message):
  return JSONResponse(
    {'error': converter.unstructure(ErrorResponse(message=message, type='invalid_request_error', code=str(status_code.value)))},
    status_code=status_code.value,
  )


async def check_model(request, model):
  if request.model == model:
    return None
  return error_response(
    HTTPStatus.NOT_FOUND,
    f"Model '{request.model}' does not exists. Try 'GET /v1/models' to see available models.\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request.",
  )


def create_logprobs(token_ids, top_logprobs, num_output_top_logprobs=None, initial_text_offset=0, *, llm):
  # Create OpenAI-style logprobs.
  logprobs = LogProbs()
  last_token_len = 0
  if num_output_top_logprobs:
    logprobs.top_logprobs = []
  for i, token_id in enumerate(token_ids):
    step_top_logprobs = top_logprobs[i]
    token_logprob = None
    if step_top_logprobs is not None:
      token_logprob = step_top_logprobs[token_id]
    token = llm.tokenizer.convert_ids_to_tokens(token_id)
    logprobs.tokens.append(token)
    logprobs.token_logprobs.append(token_logprob)
    if len(logprobs.text_offset) == 0:
      logprobs.text_offset.append(initial_text_offset)
    else:
      logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
    last_token_len = len(token)
    if num_output_top_logprobs:
      logprobs.top_logprobs.append({llm.tokenizer.convert_ids_to_tokens(i): p for i, p in step_top_logprobs.items()} if step_top_logprobs else None)
  return logprobs


def mount_to_svc(svc, llm):
  list_models.__doc__ = list_models.__doc__.replace('__model_id__', llm.llm_type)
  completions.__doc__ = completions.__doc__.replace('__model_id__', llm.llm_type)
  chat_completions.__doc__ = chat_completions.__doc__.replace('__model_id__', llm.llm_type)
  app = Starlette(
    debug=True,
    routes=[
      Route('/models', functools.partial(apply_schema(list_models, __model_id__=llm.llm_type), llm=llm), methods=['GET']),
      Route('/completions', functools.partial(apply_schema(completions, __model_id__=llm.llm_type), llm=llm), methods=['POST']),
      Route(
        '/chat/completions',
        functools.partial(
          apply_schema(
            chat_completions,
            __model_id__=llm.llm_type,
            __chat_template__=orjson.dumps(llm.config.chat_template).decode(),
            __chat_messages__=orjson.dumps(llm.config.chat_messages).decode(),
            __add_generation_prompt__=str(True) if llm.config.chat_messages is not None else str(False),
          ),
          llm=llm,
        ),
        methods=['POST'],
      ),
      Route('/schema', endpoint=lambda req: schemas.OpenAPIResponse(req), include_in_schema=False),
    ],
  )
  svc.mount_asgi_app(app, path='/v1')
  return append_schemas(svc, schemas.get_schema(routes=app.routes, mount_path='/v1'))


# GET /v1/models
@add_schema_definitions
def list_models(_, llm):
  return JSONResponse(converter.unstructure(ModelList(data=[ModelCard(id=llm.llm_type)])), status_code=HTTPStatus.OK.value)


# POST /v1/chat/completions
@add_schema_definitions
async def chat_completions(req, llm):
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
  if err_check is not None:
    return err_check

  if request.logit_bias is not None and len(request.logit_bias) > 0:
    return error_response(HTTPStatus.BAD_REQUEST, "'logit_bias' is not yet supported.")

  model_name, request_id = request.model, gen_random_uuid('chatcmpl')
  created_time = int(time.monotonic())
  prompt = llm.tokenizer.apply_chat_template(
    request.messages,
    tokenize=False,
    chat_template=request.chat_template if request.chat_template != 'None' else None,
    add_generation_prompt=request.add_generation_prompt,
  )
  logger.debug('Prompt: %r', prompt)
  config = llm.config.compatible_options(request)

  def get_role() -> str:
    return request.messages[-1]['role'] if not request.add_generation_prompt else 'assistant'  # TODO: Support custom role here.

  try:
    result_generator = llm.generate_iterator(prompt, request_id=request_id, **config)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')

  def create_stream_response_json(index, text, finish_reason=None, usage=None):
    response = ChatCompletionStreamResponse(
      id=request_id,
      created=created_time,
      model=model_name,
      choices=[ChatCompletionResponseStreamChoice(index=index, delta=Delta(content=text), finish_reason=finish_reason)],
    )
    if usage is not None:
      response.usage = usage
    return jsonify_attr(response)

  async def completion_stream_generator():
    # first chunk with role
    role = get_role()
    for i in range(config['n']):
      yield f'data: {jsonify_attr(ChatCompletionStreamResponse(id=request_id, created=created_time, choices=[ChatCompletionResponseStreamChoice(index=i, delta=Delta(role=role), finish_reason=None)], model=model_name))}\n\n'

    if request.echo:
      last_message, last_content = request.messages[-1], ''
      if last_message.get('content') and last_message.get('role') == role:
        last_content = last_message['content']
      if last_content:
        for i in range(config['n']):
          yield f'data: {jsonify_attr(ChatCompletionStreamResponse(id=request_id, created=created_time, choices=[ChatCompletionResponseStreamChoice(index=i, delta=Delta(content=last_content), finish_reason=None)], model=model_name))}\n\n'

    previous_num_tokens = [0] * config['n']
    finish_reason_sent = [False] * config['n']
    async for res in result_generator:
      for output in res.outputs:
        if finish_reason_sent[output.index]:
          continue
        yield f'data: {create_stream_response_json(output.index, output.text)}\n\n'
        previous_num_tokens[output.index] += len(output.token_ids)
        if output.finish_reason is not None:
          prompt_tokens = len(res.prompt_token_ids)
          usage = UsageInfo(prompt_tokens, previous_num_tokens[i], prompt_tokens + previous_num_tokens[i])
          yield f'data: {create_stream_response_json(output.index, "", output.finish_reason, usage)}\n\n'
          finish_reason_sent[output.index] = True
    yield 'data: [DONE]\n\n'

  try:
    # Streaming case
    if request.stream:
      return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')
    # Non-streaming case
    final_result, texts, token_ids = None, [[]] * config['n'], [[]] * config['n']
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
      outputs=[output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index]) for output in final_result.outputs]
    )

    role = get_role()
    choices = [
      ChatCompletionResponseChoice(index=output.index, message=ChatMessage(role=role, content=output.text), finish_reason=output.finish_reason)
      for output in final_result.outputs
    ]
    if request.echo:
      last_message, last_content = request.messages[-1], ''
      if last_message.get('content') and last_message.get('role') == role:
        last_content = last_message['content']
      for choice in choices:
        full_message = last_content + choice.message.content
        choice.message.content = full_message

    num_prompt_tokens = len(final_result.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_result.outputs)
    usage = UsageInfo(num_prompt_tokens, num_generated_tokens, num_prompt_tokens + num_generated_tokens)
    response = ChatCompletionResponse(id=request_id, created=created_time, model=model_name, usage=usage, choices=choices)
    return JSONResponse(converter.unstructure(response), status_code=HTTPStatus.OK.value)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')


# POST /v1/completions
@add_schema_definitions
async def completions(req, llm):
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
  if err_check is not None:
    return err_check

  # OpenAI API supports echoing the prompt when max_tokens is 0.
  echo_without_generation = request.echo and request.max_tokens == 0
  if echo_without_generation:
    request.max_tokens = 1  # XXX: Hack to make sure we get the prompt back.

  if request.suffix is not None:
    return error_response(HTTPStatus.BAD_REQUEST, "'suffix' is not yet supported.")
  if request.logit_bias is not None and len(request.logit_bias) > 0:
    return error_response(HTTPStatus.BAD_REQUEST, "'logit_bias' is not yet supported.")

  if not request.prompt:
    return error_response(HTTPStatus.BAD_REQUEST, 'Please provide a prompt.')
  prompt = request.prompt
  # TODO: Support multiple prompts

  model_name, request_id = request.model, gen_random_uuid('cmpl')
  created_time = int(time.monotonic())
  config = llm.config.compatible_options(request)

  try:
    result_generator = llm.generate_iterator(prompt, request_id=request_id, **config)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')

  # best_of != n then we don't stream
  # TODO: support use_beam_search
  stream = request.stream and (config['best_of'] is None or config['n'] == config['best_of'])

  def create_stream_response_json(index, text, logprobs=None, finish_reason=None, usage=None):
    response = CompletionStreamResponse(
      id=request_id,
      created=created_time,
      model=model_name,
      choices=[CompletionResponseStreamChoice(index=index, text=text, logprobs=logprobs, finish_reason=finish_reason)],
    )
    if usage:
      response.usage = usage
    return jsonify_attr(response)

  async def completion_stream_generator():
    previous_num_tokens = [0] * config['n']
    previous_texts = [''] * config['n']
    previous_echo = [False] * config['n']
    async for res in result_generator:
      for output in res.outputs:
        i = output.index
        delta_text = output.text
        token_ids = output.token_ids
        logprobs = None
        top_logprobs = None
        if request.logprobs is not None:
          top_logprobs = output.logprobs[previous_num_tokens[i] :]

        if request.echo and not previous_echo[i]:
          if not echo_without_generation:
            delta_text = res.prompt + delta_text
            token_ids = res.prompt_token_ids + token_ids
            if top_logprobs:
              top_logprobs = res.prompt_logprobs + top_logprobs
          else:
            delta_text = res.prompt
            token_ids = res.prompt_token_ids
            if top_logprobs:
              top_logprobs = res.prompt_logprobs
          previous_echo[i] = True
        if request.logprobs is not None:
          logprobs = create_logprobs(output.token_ids, output.logprobs[previous_num_tokens[i] :], request.logprobs, len(previous_texts[i]), llm=llm)
        previous_num_tokens[i] += len(output.token_ids)
        previous_texts[i] += output.text
        yield f'data: {create_stream_response_json(index=i, text=output.text, logprobs=logprobs, finish_reason=output.finish_reason)}\n\n'
        if output.finish_reason is not None:
          logprobs = LogProbs() if request.logprobs is not None else None
          prompt_tokens = len(res.prompt_token_ids)
          usage = UsageInfo(prompt_tokens, previous_num_tokens[i], prompt_tokens + previous_num_tokens[i])
          yield f'data: {create_stream_response_json(i, "", logprobs, output.finish_reason, usage)}\n\n'
    yield 'data: [DONE]\n\n'

  try:
    # Streaming case
    if stream:
      return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')
    # Non-streaming case
    final_result, texts, token_ids = None, [[]] * config['n'], [[]] * config['n']
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
      outputs=[output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index]) for output in final_result.outputs]
    )

    choices = []
    prompt_token_ids = final_result.prompt_token_ids
    prompt_logprobs = final_result.prompt_logprobs
    prompt_text = final_result.prompt
    for output in final_result.outputs:
      logprobs = None
      if request.logprobs is not None:
        if not echo_without_generation:
          token_ids, top_logprobs = output.token_ids, output.logprobs
          if request.echo:
            token_ids, top_logprobs = prompt_token_ids + token_ids, prompt_logprobs + top_logprobs
        else:
          token_ids, top_logprobs = prompt_token_ids, prompt_logprobs
        logprobs = create_logprobs(token_ids, top_logprobs, request.logprobs, llm=llm)
      if not echo_without_generation:
        output_text = output.text
        if request.echo:
          output_text = prompt_text + output_text
      else:
        output_text = prompt_text
      choice_data = CompletionResponseChoice(index=output.index, text=output_text, logprobs=logprobs, finish_reason=output.finish_reason)
      choices.append(choice_data)

    num_prompt_tokens = len(final_result.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_result.outputs)
    usage = UsageInfo(num_prompt_tokens, num_generated_tokens, num_prompt_tokens + num_generated_tokens)
    response = CompletionResponse(id=request_id, created=created_time, model=model_name, usage=usage, choices=choices)
    return JSONResponse(converter.unstructure(response), status_code=HTTPStatus.OK.value)
  except Exception as err:
    traceback.print_exc()
    logger.error('Error generating completion: %s', err)
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, f'Exception: {err!s} (check server log)')
