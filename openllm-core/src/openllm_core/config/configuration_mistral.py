from __future__ import annotations

import openllm_core

SINST_KEY, EINST_KEY, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'


class MistralConfig(openllm_core.LLMConfig):
  """Mistral-7B-v0.1 is a small, yet powerful model adaptable to many use-cases. It is released under Apache 2.0 licence, and it's easy to deploy on any cloud

  It currently contains a foundation model and a instruct-finetuned model.

  Refer to [Mistral's HuggingFace page](https://huggingface.co/docs/transformers/main/model_doc/mistral)
  for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://mistral.ai',
    'architecture': 'MistralForCausalLM',
    'add_generation_prompt': True,
    'default_id': 'mistralai/Mistral-7B-Instruct-v0.1',
    'serialisation': 'safetensors',
    'backend': ('pt', 'vllm'),
    # NOTE: see https://docs.mistral.ai/usage/guardrailing/ and https://docs.mistral.ai/llm/mistral-instruct-v0.1
    'model_ids': [
      'HuggingFaceH4/zephyr-7b-alpha',
      'HuggingFaceH4/zephyr-7b-beta',
      'mistralai/Mistral-7B-Instruct-v0.1',
      'mistralai/Mistral-7B-v0.1',
    ],
    'fine_tune_strategies': (
      {'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'},
    ),
  }

  class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5

  @property
  def template(self) -> str:
    return '''{start_key}{start_inst} {system_message} {instruction} {end_inst}\n'''.format(
      start_inst=SINST_KEY,
      end_inst=EINST_KEY,
      start_key=BOS_TOKEN,
      system_message='{system_message}',
      instruction='{instruction}',
    )

  # NOTE: https://docs.mistral.ai/usage/guardrailing/
  @property
  def system_message(self) -> str:
    return '''Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.'''
