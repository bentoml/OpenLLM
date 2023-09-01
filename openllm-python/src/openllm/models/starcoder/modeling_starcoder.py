from __future__ import annotations
import logging
import typing as t

import bentoml
import openllm
from openllm.utils import generate_labels
from openllm_core.config.configuration_starcoder import EOD
from openllm_core.config.configuration_starcoder import FIM_MIDDLE
from openllm_core.config.configuration_starcoder import FIM_PAD
from openllm_core.config.configuration_starcoder import FIM_PREFIX
from openllm_core.config.configuration_starcoder import FIM_SUFFIX
if t.TYPE_CHECKING: import transformers

class StarCoder(openllm.LLM['transformers.GPTBigCodeForCausalLM', 'transformers.GPT2TokenizerFast']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, 'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}, {}

  def import_model(self, *args: t.Any, trust_remote_code: bool = False, **attrs: t.Any) -> bentoml.Model:
    import torch
    import transformers
    torch_dtype, device_map = attrs.pop('torch_dtype', torch.float16), attrs.pop('device_map', 'auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **self.llm_parameters[-1])
    tokenizer.add_special_tokens({'additional_special_tokens': [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD], 'pad_token': EOD})
    model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch_dtype, device_map=device_map, **attrs)
    try:
      return bentoml.transformers.save_model(self.tag, model, custom_objects={'tokenizer': tokenizer}, labels=generate_labels(self))
    finally:
      torch.cuda.empty_cache()

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    with torch.inference_mode():
      # eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"), # NOTE: this is for finetuning starcoder
      # NOTE: support fine-tuning starcoder
      result_tensor = self.model.generate(self.tokenizer.encode(prompt, return_tensors='pt').to(self.device),
                                          do_sample=True,
                                          pad_token_id=self.tokenizer.eos_token_id,
                                          generation_config=self.config.model_construct_env(**attrs).to_generation_config())
      # TODO: We will probably want to return the tokenizer here so that we can manually process this
      # return (skip_special_tokens=False, clean_up_tokenization_spaces=False))
      return self.tokenizer.batch_decode(result_tensor[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

  def generate_one(self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any) -> list[dict[t.Literal['generated_text'], str]]:
    max_new_tokens, encoded_inputs = preprocess_generate_kwds.pop('max_new_tokens', 200), self.tokenizer(prompt, return_tensors='pt').to(self.device)
    src_len, stopping_criteria = encoded_inputs['input_ids'].shape[1], preprocess_generate_kwds.pop('stopping_criteria', openllm.StoppingCriteriaList([]))
    stopping_criteria.append(openllm.StopSequenceCriteria(stop, self.tokenizer))
    result = self.tokenizer.decode(self.model.generate(encoded_inputs['input_ids'], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)[0].tolist()[src_len:])
    # Inference API returns the stop sequence
    for stop_seq in stop:
      if result.endswith(stop_seq): result = result[:-len(stop_seq)]
    return [{'generated_text': result}]
