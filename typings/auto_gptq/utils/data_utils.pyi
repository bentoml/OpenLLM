from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from torch import LongTensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

def make_data_block(samples: Dict[str, List[str]], prompt_col_name: str, label_col_name: str, tokenizer: PreTrainedTokenizer, preprocess_fn: Optional[Callable] = ..., sample_max_len: int = ..., block_max_len: int = ..., add_eos_token: bool = ..., truncate_prompt: bool = ..., merge_prompt_label: bool = ...) -> Dict[str, List[LongTensor]]: ...
def collate_data(blocks: List[Dict[str, List[List[int]]]], pad_token_id: int) -> Dict[str, LongTensor]: ...
def get_dataloader(data_path_or_name: str, prompt_col_name: str, label_col_name: str, tokenizer: PreTrainedTokenizer, load_fn: Optional[Callable] = ..., preprocess_fn: Optional[Callable] = ..., num_samples: int = ..., sample_max_len: int = ..., block_max_len: int = ..., add_eos_token: bool = ..., truncate_prompt: bool = ..., merge_prompt_label: bool = ..., load_fn_kwargs: Optional[dict] = ..., preprocess_fn_kwargs: Optional[dict] = ..., **kwargs) -> DataLoader: ...
