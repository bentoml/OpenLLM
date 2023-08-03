from typing import List
from typing import Optional
from typing import Union

from _typeshed import Incomplete
from transformers import PreTrainedTokenizer as PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast as PreTrainedTokenizerFast
from vllm.engine.arg_utils import EngineArgs as EngineArgs
from vllm.engine.llm_engine import LLMEngine as LLMEngine
from vllm.outputs import RequestOutput as RequestOutput
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils import Counter as Counter

class LLM:
    llm_engine: Incomplete
    request_counter: Incomplete
    def __init__(self, model: str, tokenizer: Optional[str] = ..., tokenizer_mode: str = ..., trust_remote_code: bool = ..., tensor_parallel_size: int = ..., dtype: str = ..., seed: int = ..., **kwargs) -> None: ...
    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]: ...
    def set_tokenizer(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> None: ...
    def generate(self, prompts: Optional[Union[str, List[str]]] = ..., sampling_params: Optional[SamplingParams] = ..., prompt_token_ids: Optional[List[List[int]]] = ..., use_tqdm: bool = ...) -> List[RequestOutput]: ...
