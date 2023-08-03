from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import torch
from tqdm.auto import tqdm

class Disabledtqdm(tqdm):
    def __init__(self, *args, **kwargs) -> None: ...

def hf_model_weights_iterator(model_name_or_path: str, cache_dir: Optional[str] = ..., use_np_cache: bool = ...) -> Iterator[Tuple[str, torch.Tensor]]: ...
def load_tensor_parallel_weights(param: torch.Tensor, loaded_weight: torch.Tensor, param_name: str, column_parallel_weight_names: List[str], row_parallel_weight_names: List[str], tensor_model_parallel_rank: int) -> None: ...
def initialize_dummy_weights(model: torch.nn.Module, low: float = ..., high: float = ...) -> None: ...
