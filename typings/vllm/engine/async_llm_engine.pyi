from typing import List
from typing import Optional

from _typeshed import Incomplete
from vllm.config import ModelConfig as ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine as LLMEngine
from vllm.engine.ray_utils import initialize_cluster as initialize_cluster
from vllm.engine.ray_utils import ray as ray
from vllm.logger import init_logger as init_logger
from vllm.outputs import RequestOutput as RequestOutput
from vllm.sampling_params import SamplingParams as SamplingParams

logger: Incomplete
TIMEOUT_TO_PREVENT_DEADLOCK: int

class AsyncLLMEngine:
    worker_use_ray: Incomplete
    engine_use_ray: Incomplete
    log_requests: Incomplete
    engine: Incomplete
    request_outputs: Incomplete
    request_events: Incomplete
    is_engine_running: bool
    kicking_request_id: Incomplete
    def __init__(self, worker_use_ray: bool, engine_use_ray: bool, *args, log_requests: bool = ..., **kwargs) -> None: ...
    async def engine_step(self, kicking_request_id: Optional[str] = ...): ...
    async def generate(self, prompt: Optional[str], sampling_params: SamplingParams, request_id: str, prompt_token_ids: Optional[List[int]] = ...) -> RequestOutput: ...
    async def abort(self, request_id: str) -> None: ...
    async def get_model_config(self) -> ModelConfig: ...
    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> AsyncLLMEngine: ...
