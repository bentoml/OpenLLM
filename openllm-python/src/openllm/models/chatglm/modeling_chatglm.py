from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING:
  import transformers

class ChatGLM(openllm.LLM['transformers.PreTrainedModel', 'transformers.PreTrainedTokenizerFast']):
  __openllm_internal__ = True

  def generate(self, prompt: str, **attrs: t.Any) -> tuple[str, list[tuple[str, str]]]:
    import torch
    with torch.inference_mode():
      self.model.eval()
      # Only use half precision if the model is not yet quantized
      if self.config.use_half_precision: self.model.half()
      return self.model.chat(self.tokenizer, prompt, generation_config=self.config.model_construct_env(**attrs).to_generation_config())
