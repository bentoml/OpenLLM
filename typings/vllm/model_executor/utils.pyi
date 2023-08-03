from vllm.model_executor.parallel_utils.parallel_state import (
    model_parallel_is_initialized as model_parallel_is_initialized,
)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    model_parallel_cuda_manual_seed as model_parallel_cuda_manual_seed,
)

def set_random_seed(seed: int) -> None: ...
