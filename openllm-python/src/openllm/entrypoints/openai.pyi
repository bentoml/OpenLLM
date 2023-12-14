from http import HTTPStatus
from typing import Dict, List, Optional, Union

from attr import AttrsInstance
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bentoml import Service
from openllm_core._typing_compat import M, T

from .._llm import LLM
from ..protocol.openai import ChatCompletionRequest, CompletionRequest, LogProbs

def mount_to_svc(svc: Service, llm: LLM[M, T]) -> Service: ...
def jsonify_attr(obj: AttrsInstance) -> str: ...
def error_response(status_code: HTTPStatus, message: str) -> JSONResponse: ...
async def check_model(request: Union[CompletionRequest, ChatCompletionRequest], model: str) -> Optional[JSONResponse]: ...
def create_logprobs(
  token_ids: List[int],
  top_logprobs: List[Dict[int, float]],  #
  num_output_top_logprobs: Optional[int] = ...,
  initial_text_offset: int = ...,
  *,
  llm: LLM[M, T],
) -> LogProbs: ...
def list_models(req: Request, llm: LLM[M, T]) -> Response: ...
async def chat_completions(req: Request, llm: LLM[M, T]) -> Response: ...
async def completions(req: Request, llm: LLM[M, T]) -> Response: ...
