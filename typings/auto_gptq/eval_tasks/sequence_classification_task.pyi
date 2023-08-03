from typing import Dict
from typing import List
from typing import Optional

from _typeshed import Incomplete
from transformers import GenerationConfig
from transformers import PreTrainedTokenizer

from ._base import BaseTask

class SequenceClassificationTask(BaseTask):
    classes: Incomplete
    max_new_tokens: Incomplete
    def __init__(self, model, tokenizer: PreTrainedTokenizer, classes: List[str], data_name_or_path: str, prompt_col_name: str, label_col_name: str, device: Optional[str] = ..., **kwargs) -> None: ...
    def run(self, generation_config: Optional[GenerationConfig] = ...) -> Dict[str, float]: ...
