from __future__ import annotations
import typing as t

import openllm
from openllm_core._prompt import process_prompt
from openllm_core.config.configuration_flan_t5 import DEFAULT_PROMPT_TEMPLATE
if t.TYPE_CHECKING: import transformers

class FlaxFlanT5(openllm.LLM['transformers.FlaxT5ForConditionalGeneration', 'transformers.T5TokenizerFast']):
  __openllm_internal__ = True

  def sanitize_parameters(self,
                          prompt: str,
                          max_new_tokens: int | None = None,
                          temperature: float | None = None,
                          top_k: int | None = None,
                          top_p: float | None = None,
                          repetition_penalty: float | None = None,
                          decoder_start_token_id: int | None = None,
                          use_default_prompt_template: bool = True,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    if decoder_start_token_id is None: decoder_start_token_id = 0
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'decoder_start_token_id': decoder_start_token_id
    }, {}

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    # NOTE: decoder_start_token_id is extracted from https://huggingface.co/google/flan-t5-small/tree/main as it is required for encoder-decoder generation.
    decoder_start_token_id = attrs.pop('decoder_start_token_id', 0)
    return self.tokenizer.batch_decode(self.model.generate(self.tokenizer(prompt, return_tensors='np')['input_ids'],
                                                           do_sample=True,
                                                           generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
                                                           decoder_start_token_id=decoder_start_token_id).sequences,
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True)
