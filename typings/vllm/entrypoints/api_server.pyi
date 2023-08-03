from _typeshed import Incomplete
from fastapi import Request as Request
from fastapi.responses import Response
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLMEngine
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils import random_uuid as random_uuid

TIMEOUT_KEEP_ALIVE: int
TIMEOUT_TO_PREVENT_DEADLOCK: int
app: Incomplete

async def generate(request: Request) -> Response: ...
