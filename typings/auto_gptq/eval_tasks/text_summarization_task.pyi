from typing import Dict
from typing import Optional

from transformers import GenerationConfig

from ._base import BaseTask

class TextSummarizationTask(BaseTask):
    def __init__(self, model, tokenizer, data_name_or_path: str, prompt_col_name: str, label_col_name: str, device: Optional[str] = ..., **kwargs) -> None: ...
    def run(self, generation_config: Optional[GenerationConfig] = ...) -> Dict[str, float]: ...
