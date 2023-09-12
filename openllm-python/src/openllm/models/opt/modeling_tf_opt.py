from __future__ import annotations
import typing as t

import bentoml
import openllm
from openllm_core.utils import generate_labels
if t.TYPE_CHECKING: import transformers

class TFOPT(openllm.LLM['transformers.TFOPTForCausalLM', 'transformers.GPT2Tokenizer']):
  __openllm_internal__ = True

  def import_model(self, *args: t.Any, trust_remote_code: bool = False, **attrs: t.Any) -> bentoml.Model:
    import transformers
    config, tokenizer = transformers.AutoConfig.from_pretrained(self.model_id), transformers.AutoTokenizer.from_pretrained(self.model_id, **self.llm_parameters[-1])
    tokenizer.pad_token_id = config.pad_token_id
    return bentoml.transformers.save_model(self.tag,
                                           transformers.TFOPTForCausalLM.from_pretrained(self.model_id, trust_remote_code=trust_remote_code, **attrs),
                                           custom_objects={'tokenizer': tokenizer},
                                           labels=generate_labels(self))

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors='tf'),
                                                           do_sample=True,
                                                           generation_config=self.config.model_construct_env(**attrs).to_generation_config()),
                                       skip_special_tokens=True)
