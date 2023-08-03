from typing import List
from typing import Tuple
from typing import Union

from _typeshed import Incomplete
from transformers import PreTrainedTokenizer as PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast
from vllm.logger import init_logger as init_logger

logger: Incomplete

def get_tokenizer(tokenizer_name: str, *args, tokenizer_mode: str = ..., trust_remote_code: bool = ..., **kwargs) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]: ...
def detokenize_incrementally(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], prev_output_tokens: List[str], new_token_id: int, skip_special_tokens: bool) -> Tuple[str, str]: ...
