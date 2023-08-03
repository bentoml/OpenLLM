from typing import List
from typing import Optional
from typing import Union

from torch import LongTensor
from transformers import PreTrainedTokenizer

def postprocess_generation_ids(input_ids: LongTensor, output_ids: LongTensor, num_return_sequences: int, tokenizer: Optional[PreTrainedTokenizer] = ..., pad_token_ids: Optional[int] = ...) -> List[List[Union[str, List[int]]]]: ...
