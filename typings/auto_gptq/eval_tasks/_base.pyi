import abc
from typing import Dict
from typing import Optional
from typing import Union

from _typeshed import Incomplete
from transformers import PreTrainedModel as PreTrainedModel
from transformers import PreTrainedTokenizer as PreTrainedTokenizer

from ..modeling import BaseGPTQForCausalLM as BaseGPTQForCausalLM
from ..utils.data_utils import get_dataloader as get_dataloader

class BaseTask(metaclass=abc.ABCMeta):
    model: Incomplete
    tokenizer: Incomplete
    dl: Incomplete
    device: Incomplete
    def __init__(self, model: Union[BaseGPTQForCausalLM, PreTrainedModel], tokenizer: PreTrainedTokenizer, data_name_or_path: str, prompt_col_name: str, label_col_name: str, device: Optional[str] = ..., **kwargs) -> None: ...
    def run(self, **predict_kwargs) -> Dict[str, float]: ...
