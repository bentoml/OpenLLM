from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING:
  import transformers

class StableLM(openllm.LLM['transformers.GPTNeoXForCausalLM', 'transformers.GPTNeoXTokenizerFast']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}, {}

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    with torch.inference_mode():
      return [
          self.tokenizer.decode(self.model.generate(**self.tokenizer(prompt, return_tensors='pt').to(self.device),
                                                    do_sample=True,
                                                    generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
                                                    pad_token_id=self.tokenizer.eos_token_id,
                                                    stopping_criteria=openllm.StoppingCriteriaList([openllm.StopOnTokens()]))[0],
                                skip_special_tokens=True)
      ]
