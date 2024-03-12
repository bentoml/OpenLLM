from __future__ import annotations

import openllm, traceback, logging, time, pydantic, typing as t
from openllm_core.exceptions import ModelNotFound, OpenLLMException, ValidationError
from openllm_core.utils import gen_random_uuid
from openllm_core.protocol.openai import (
  ChatCompletionRequest,
  CompletionRequest,
  CompletionResponse,
  ChatCompletionResponse,
  ErrorResponse,
  NotSupportedError,
  LogProbs,
)


logger = logging.getLogger(__name__)


class Error(pydantic.BaseModel):
  error: ErrorResponse


def error_response(exception: type[OpenLLMException], message: str) -> ErrorResponse:
  return Error(
    error=ErrorResponse(message=message, type=str(exception.__qualname__), code=str(exception.error_code.value))
  )


class OpenAI:
  @staticmethod
  async def chat_completions(self: openllm.LLM, **request) -> t.AsyncGenerator[ChatCompletionResponse, None]:
    try:
      request = ChatCompletionRequest.model_construct(**request)
    except Exception:
      traceback.print_exc()
      return OpenAI.create_error_response(f'Invalid request: {traceback.format_exc()}')

    if request.logit_bias is not None and len(request.logit_bias) > 0:
      return OpenAI.create_error_response("'logit_bias' is not supported .", NotSupportedError)

    error = await self._check_model(request)
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
    if request.model != self.model_id:
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
