from http import HTTPStatus
from typing import Dict
from typing import List
from typing import Optional

from _typeshed import Incomplete
from fastapi import Request as Request
from fastapi.responses import JSONResponse
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionRequest as ChatCompletionRequest
from vllm.entrypoints.openai.protocol import ChatCompletionResponse as ChatCompletionResponse
from vllm.entrypoints.openai.protocol import ChatCompletionResponseChoice as ChatCompletionResponseChoice
from vllm.entrypoints.openai.protocol import ChatCompletionResponseStreamChoice as ChatCompletionResponseStreamChoice
from vllm.entrypoints.openai.protocol import ChatCompletionStreamResponse as ChatCompletionStreamResponse
from vllm.entrypoints.openai.protocol import ChatMessage as ChatMessage
from vllm.entrypoints.openai.protocol import CompletionRequest as CompletionRequest
from vllm.entrypoints.openai.protocol import CompletionResponse as CompletionResponse
from vllm.entrypoints.openai.protocol import CompletionResponseChoice as CompletionResponseChoice
from vllm.entrypoints.openai.protocol import CompletionResponseStreamChoice as CompletionResponseStreamChoice
from vllm.entrypoints.openai.protocol import CompletionStreamResponse as CompletionStreamResponse
from vllm.entrypoints.openai.protocol import DeltaMessage as DeltaMessage
from vllm.entrypoints.openai.protocol import ErrorResponse as ErrorResponse
from vllm.entrypoints.openai.protocol import LogProbs as LogProbs
from vllm.entrypoints.openai.protocol import ModelCard as ModelCard
from vllm.entrypoints.openai.protocol import ModelList as ModelList
from vllm.entrypoints.openai.protocol import ModelPermission as ModelPermission
from vllm.entrypoints.openai.protocol import UsageInfo as UsageInfo
from vllm.logger import init_logger as init_logger
from vllm.outputs import RequestOutput as RequestOutput
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer as get_tokenizer
from vllm.utils import random_uuid as random_uuid

TIMEOUT_KEEP_ALIVE: int
logger: Incomplete
served_model: Incomplete
app: Incomplete

def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse: ...
async def validation_exception_handler(request, exc): ...
async def check_model(request) -> Optional[JSONResponse]: ...
async def get_gen_prompt(request) -> str: ...
async def check_length(request, prompt): ...
async def show_available_models(): ...
def create_logprobs(token_ids: List[int], id_logprobs: List[Dict[int, float]], initial_text_offset: int = ...) -> LogProbs: ...
async def create_chat_completion(raw_request: Request): ...
async def create_completion(raw_request: Request): ...
