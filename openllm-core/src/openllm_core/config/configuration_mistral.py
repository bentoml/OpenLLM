from __future__ import annotations

import openllm_core, typing as t

if t.TYPE_CHECKING:
  from openllm_core._schemas import MessageParam

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
    'default_id': 'mistralai/Mistral-7B-Instruct-v0.1',
    'serialisation': 'safetensors',
    'model_ids': [
      'HuggingFaceH4/zephyr-7b-alpha',
      'HuggingFaceH4/zephyr-7b-beta',
      'mistralai/Mistral-7B-Instruct-v0.2',
      'mistralai/Mistral-7B-Instruct-v0.1',
      'mistralai/Mistral-7B-v0.1',
    ],
    'fine_tune_strategies': ({'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'},),
  }

  class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5

  # NOTE: see https://docs.mistral.ai/usage/guardrailing/ and https://docs.mistral.ai/llm/mistral-instruct-v0.1
  @property
  def template(self) -> str:
    return """{start_key}{start_inst} {system_message} {instruction} {end_inst}\n""".format(
      start_inst=SINST_KEY, end_inst=EINST_KEY, start_key=BOS_TOKEN, system_message='{system_message}', instruction='{instruction}'
    )

  # NOTE: https://docs.mistral.ai/usage/guardrailing/
  @property
  def system_message(self) -> str:
    return """Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."""

  @property
  def chat_template(self) -> str:
    return repr(
      "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    )

  @property
  def chat_messages(self) -> list[MessageParam]:
    from openllm_core._schemas import MessageParam

    return [
      MessageParam(role='user', content='What is your favourite condiment?'),
      MessageParam(
        role='assistant',
        content="Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
      ),
      MessageParam(role='user', content='Do you have mayonnaise recipes?'),
    ]
