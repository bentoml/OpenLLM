from __future__ import annotations

import openllm, traceback, logging, time, pydantic, typing as t
from openllm_core.exceptions import ModelNotFound, OpenLLMException, ValidationError
from openllm_core.utils import gen_random_uuid
from openllm_core.protocol.openai import (
  ChatCompletionRequest,
  ChatCompletionResponseChoice,
  ChatCompletionResponseStreamChoice,
  ChatCompletionStreamResponse,
  ChatMessage,
  CompletionRequest,
  CompletionResponse,
  ChatCompletionResponse,
  Delta,
  ErrorResponse,
  NotSupportedError,
  LogProbs,
  UsageInfo,
)

if t.TYPE_CHECKING:
  from vllm import RequestOutput


logger = logging.getLogger(__name__)


class Error(pydantic.BaseModel):
  error: ErrorResponse


def error_response(exception: type[OpenLLMException], message: str) -> ErrorResponse:
  return Error(
    error=ErrorResponse(message=message, type=str(exception.__qualname__), code=str(exception.error_code.value))
  )


class OpenAI:
  @staticmethod
  async def chat_completions(
    self: openllm.LLM, request: ChatCompletionRequest
  ) -> t.AsyncGenerator[ChatCompletionResponse, None]:
    if request.logit_bias is not None and len(request.logit_bias) > 0:
      return OpenAI.create_error_response("'logit_bias' is not supported .", NotSupportedError)

    error = await OpenAI._check_model(self, request)
    if error is not None:
      return error

    try:
      prompt = self._tokenizer.apply_chat_template(
        conversation=request.messages,
        tokenize=False,
        add_generation_prompt=request.add_generation_prompt,
        chat_template=request.chat_template if request.chat_template != 'None' else None,
      )
    except Exception as e:
      traceback.print_exc()
      return OpenAI.create_error_response(f'Failed to apply chat template: {e}')

    model_name, request_id = request.model, gen_random_uuid('chatcmpl')
    created_time = int(time.monotonic())

    generator = self.generate_iterator(prompt, request_id=request_id, **request.model_dump())
    if request.stream:
      return OpenAI.chat_completion_stream_generator(request, self, model_name, request_id, created_time, generator)

    try:
      return await OpenAI.chat_completion_full_generator(
        request, self, model_name, request_id, created_time, generator
      )
    except Exception as err:
      traceback.print_exc()
      return OpenAI.create_error_response(str(err))

  @staticmethod
  async def chat_completion_stream_generator(
    request: ChatCompletionRequest,
    llm: openllm.LLM,
    model_name: str,
    request_id: str,
    created_time: int,
    generator: t.AsyncIterator[RequestOutput],
  ):
    first_iteration = True

    previous_texts = [''] * llm.config['n']
    previous_num_tokens = [0] * llm.config['n']
    finish_reason_sent = [False] * llm.config['n']
    try:
      async for request_output in generator:
        if first_iteration:
          role = OpenAI.get_chat_role(request)
          for i in range(llm.config['n']):
            choice_data = ChatCompletionResponseStreamChoice(
              index=i, delta=Delta(role=role), logprobs=None, finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
              id=request_id, created=created_time, choices=[choice_data], model=model_name
            )
            data = chunk.model_dump_json(exclude_unset=True)
          yield f'data: {data}\n\n'

          if request.echo:
            last_msg_content = ''
            if (
              request.messages
              and isinstance(request.messages, list)
              and request.messages[-1].get('content')
              and request.messages[-1].get('role') == role
            ):
              last_msg_content = request.messages[-1]['content']

            if last_msg_content:
              for i in range(request.n):
                choice_data = ChatCompletionResponseStreamChoice(
                  index=i, delta=Delta(content=last_msg_content), finish_reason=None
                )
                chunk = ChatCompletionStreamResponse(
                  id=request_id, created=created_time, choices=[choice_data], logprobs=None, model=model_name
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f'data: {data}\n\n'
          first_iteration = False

        for output in request_output.outputs:
          i = output.index

          if finish_reason_sent[i]:
            continue

          delta_token_ids = output.token_ids[previous_num_tokens[i] :]
          top_logprobs = output.logprobs[previous_num_tokens[i] :] if output.logprobs else None
          logprobs = None

          if request.logprobs:
            logprobs = OpenAI._create_logprobs(
              llm,
              token_ids=delta_token_ids,
              top_logprobs=top_logprobs,
              num_output_top_logprobs=request.logprobs,
              initial_text_offset=len(previous_texts[i]),
            )

          delta_text = output.text[len(previous_texts[i]) :]
          previous_texts[i] = output.text
          previous_num_tokens[i] = len(output.token_ids)
          if output.finish_reason is None:
            # Send token-by-token response for each request.n
            choice_data = ChatCompletionResponseStreamChoice(
              index=i, delta=Delta(content=delta_text), logprobs=logprobs, finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(
              id=request_id, created=created_time, choices=[choice_data], model=model_name
            )
            data = chunk.model_dump_json(exclude_unset=True)
            yield f'data: {data}\n\n'
          else:
            # Send the finish response for each request.n only once
            prompt_tokens = len(request_output.prompt_token_ids)
            final_usage = UsageInfo(
              prompt_tokens=prompt_tokens,
              completion_tokens=previous_num_tokens[i],
              total_tokens=prompt_tokens + previous_num_tokens[i],
            )
            choice_data = ChatCompletionResponseStreamChoice(
              index=i, delta=Delta(content=delta_text), logprobs=logprobs, finish_reason=output.finish_reason
            )
            chunk = ChatCompletionStreamResponse(
              id=request_id, created=created_time, choices=[choice_data], model=model_name
            )
            if final_usage is not None:
              chunk.usage = final_usage
            data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f'data: {data}\n\n'
            finish_reason_sent[i] = True
    except ValueError as e:
      data = OpenAI.create_error_response(str(e)).model_dump_json()
      yield f'data: {data}\n\n'
    # Send the final done message after all response.n are finished
    yield 'data: [DONE]\n\n'

  @staticmethod
  async def chat_completion_full_generator(
    request: ChatCompletionRequest,
    llm: openllm.LLM,
    model_name: str,
    request_id: str,
    created_time: int,
    generator: t.AsyncIterator[RequestOutput],
  ):
    final_result: RequestOutput = None

    # TODO: Check if raw_request is disconnected or not.
    try:
      async for request_output in generator:
        final_result = request_output
    except ValueError as e:
      await llm._model.abort(request_id)
      return OpenAI.create_error_response(str(e))

    if final_result is None:
      return OpenAI.create_error_response('No result is returned.')

    choices = []
    role = OpenAI.get_chat_role(request)

    for output in final_result.outputs:
      token_ids = output.token_ids
      top_logprobs = output.logprobs

      logprobs = None
      if request.logprobs:
        logprobs = OpenAI._create_logprobs(
          llm, token_ids=token_ids, top_logprobs=top_logprobs, num_output_top_logprobs=request.logprobs
        )

      choice_data = ChatCompletionResponseChoice(
        index=output.index,
        message=ChatMessage(role=role, content=output.text),
        logprobs=logprobs,
        finish_reason=output.finish_reason,
      )
      choices.append(choice_data)

    if request.echo:
      last_msg_content = ''
      if (
        request.messages
        and isinstance(request.messages, list)
        and request.messages[-1].get('content')
        and request.messages[-1].get('role') == role
      ):
        last_msg_content = request.messages[-1]['content']

      for choice in choices:
        full_message = last_msg_content + choice.message.content
        choice.message.content = full_message

    num_prompt_tokens = len(final_result.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_result.outputs)
    usage = UsageInfo(
      prompt_tokens=num_prompt_tokens,
      completion_tokens=num_generated_tokens,
      total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
      id=request_id, created=created_time, model=model_name, choices=choices, usage=usage
    )

    return response

  # --- chat specific ---
  @staticmethod
  def get_chat_role(request: ChatCompletionRequest) -> str:
    return request.messages[-1]['role'] if not request.add_generation_prompt else 'assistant'

  @staticmethod
  async def completions(self: openllm.LLM, **request) -> t.AsyncGenerator[CompletionResponse, None]: ...

  # ---
  @staticmethod
  def create_error_response(message: str, exception: type[OpenLLMException] = ValidationError) -> ErrorResponse:
    return error_response(exception, message)

  @staticmethod
  async def _check_model(self: openllm.LLM, request: ChatCompletionRequest | CompletionRequest) -> Error | None:
    if request.model != self.llm_type:
      return error_response(
        ModelNotFound,
        f"Model '{request.model}' does not exists. Try 'GET /v1/models' to see available models.\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request.",
      )
    # TODO: support lora

  @staticmethod
  def _validate_prompt(
    self: openllm.LLM,
    request: ChatCompletionRequest | CompletionRequest,
    prompt: str | None = None,
    prompt_ids: list[int] | None = None,
  ) -> Error | None:
    if not (prompt or prompt_ids):
      raise ValueError("'prompt' or 'prompt_ids' must be provided.")
    if prompt and prompt_ids:
      raise ValueError("'prompt' and 'prompt_ids' are mutually exclusive.")
    # TODO: valudate max context length based on requested max_tokens
    #
    # input_ids = prompt_ids if prompt_ids is not None else self._tokenizer(prompt).input_ids
    # token_num = len(input_ids)
    # if request.max_tokens is None: request.max_tokens = self.engine_args['max_model_len']

  @staticmethod
  def _create_logprobs(
    self: openllm.LLM,
    token_ids: list[int],
    top_logprobs: list[dict[int, float] | None] | None = None,  #
    num_output_top_logprobs: int | None = None,
    initial_text_offset: int = 0,
  ) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    if num_output_top_logprobs:
      logprobs.top_logprobs = []
    for i, token_id in enumerate(token_ids):
      step_top_logprobs = top_logprobs[i]
      if step_top_logprobs is not None:
        token_logprob = step_top_logprobs[token_id].logprob
      else:
        token_logprob = None
      token = step_top_logprobs[token_id].decoded_token
      logprobs.tokens.append(token)
      logprobs.token_logprobs.append(token_logprob)
      if len(logprobs.text_offset) == 0:
        logprobs.text_offset.append(initial_text_offset)
      else:
        logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
      last_token_len = len(token)

      if num_output_top_logprobs:
        logprobs.top_logprobs.append(
          {p.decoded_token: p.logprob for i, p in step_top_logprobs.items()} if step_top_logprobs else None
        )
    return logprobs