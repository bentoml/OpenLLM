from typing import Dict
from typing import Optional

from ._base import BaseTask

class LanguageModelingTask(BaseTask):
    def __init__(self, model, tokenizer, data_name_or_path: str, prompt_col_name: str, label_col_name: str, device: Optional[str] = ..., **kwargs) -> None: ...
    def run(self) -> Dict[str, float]: ...
