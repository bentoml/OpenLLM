from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING: import torch, transformers
else:
  torch, transformers = openllm.utils.LazyLoader('torch', globals(), 'torch'), openllm.utils.LazyLoader('transformers', globals(), 'transformers')

class Falcon(openllm.LLM['transformers.PreTrainedModel', 'transformers.PreTrainedTokenizerBase']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    return {'torch_dtype': torch.bfloat16, 'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None}, {}

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    eos_token_id, inputs = attrs.pop('eos_token_id', self.tokenizer.eos_token_id), self.tokenizer(prompt, return_tensors='pt').to(self.device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):  # type: ignore[attr-defined]
      return self.tokenizer.batch_decode(self.model.generate(input_ids=inputs['input_ids'],
                                                             attention_mask=inputs['attention_mask'],
                                                             generation_config=self.config.model_construct_env(eos_token_id=eos_token_id, **attrs).to_generation_config()),
                                         skip_special_tokens=True)

  def generate_one(self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any) -> list[dict[t.Literal['generated_text'], str]]:
    max_new_tokens, encoded_inputs = preprocess_generate_kwds.pop('max_new_tokens', 200), self.tokenizer(prompt, return_tensors='pt').to(self.device)
    src_len, stopping_criteria = encoded_inputs['input_ids'].shape[1], preprocess_generate_kwds.pop('stopping_criteria', openllm.StoppingCriteriaList([]))
    stopping_criteria.append(openllm.StopSequenceCriteria(stop, self.tokenizer))
    result = self.tokenizer.decode(self.model.generate(encoded_inputs['input_ids'], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)[0].tolist()[src_len:])
    # Inference API returns the stop sequence
    for stop_seq in stop:
      if result.endswith(stop_seq): result = result[:-len(stop_seq)]
    return [{'generated_text': result}]
