import torch
from _typeshed import Incomplete
from vllm.model_executor.parallel_utils.parallel_state import get_all_reduce_launcher as get_all_reduce_launcher
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)

from .mappings import copy_to_tensor_model_parallel_region as copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region as gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region as reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region as scatter_to_tensor_model_parallel_region
from .random import get_cuda_rng_tracker as get_cuda_rng_tracker
from .utils import VocabUtility as VocabUtility
from .utils import divide as divide

def param_is_not_tensor_parallel_duplicate(param): ...
def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride) -> None: ...
def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor) -> None: ...
def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor) -> None: ...

class VocabParallelEmbedding(torch.nn.Module):
    num_embeddings: Incomplete
    embedding_dim: Incomplete
    padding_idx: Incomplete
    max_norm: Incomplete
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool
    tensor_model_parallel_size: Incomplete
    num_embeddings_per_partition: Incomplete
    weight: Incomplete
    def __init__(self, num_embeddings: int, embedding_dim: int, *, init_method=..., params_dtype: torch.dtype = ..., use_cpu_initialization: bool = ..., perform_initialization: bool = ...) -> None: ...
    def forward(self, input_): ...

class ColumnParallelLinear(torch.nn.Module):
    input_size: Incomplete
    output_size: Incomplete
    gather_output: Incomplete
    output_size_per_partition: Incomplete
    skip_bias_add: Incomplete
    weight: Incomplete
    master_weight: Incomplete
    bias: Incomplete
    def __init__(self, input_size, output_size, *, bias: bool = ..., gather_output: bool = ..., init_method=..., stride: int = ..., keep_master_weight_for_test: bool = ..., skip_bias_add: bool = ..., params_dtype: Incomplete | None = ..., use_cpu_initialization: bool = ..., perform_initialization: bool = ...) -> None: ...
    def forward(self, input_): ...

class RowParallelLinear(torch.nn.Module):
    input_size: Incomplete
    output_size: Incomplete
    input_is_parallel: Incomplete
    input_size_per_partition: Incomplete
    skip_bias_add: Incomplete
    weight: Incomplete
    master_weight: Incomplete
    bias: Incomplete
    weight_t: Incomplete
    def __init__(self, input_size, output_size, *, bias: bool = ..., input_is_parallel: bool = ..., init_method=..., stride: int = ..., keep_master_weight_for_test: bool = ..., skip_bias_add: bool = ..., params_dtype: Incomplete | None = ..., use_cpu_initialization: bool = ..., perform_initialization: bool = ...) -> None: ...
    def forward(self, input_): ...
