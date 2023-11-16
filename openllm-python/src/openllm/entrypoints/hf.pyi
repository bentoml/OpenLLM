from http import HTTPStatus

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bentoml import Service
from openllm_core._typing_compat import M, T

from .._llm import LLM

def mount_to_svc(svc: Service, llm: LLM[M, T]) -> Service: ...
def error_response(status_code: HTTPStatus, message: str) -> JSONResponse: ...
async def hf_agent(req: Request, llm: LLM[M, T]) -> Response: ...
def hf_adapters(req: Request, llm: LLM[M, T]) -> Response: ...
