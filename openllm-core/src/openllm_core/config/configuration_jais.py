from __future__ import annotations
import typing as t

import openllm_core

if t.TYPE_CHECKING:
  pass

INSTRUCTION_KEY = '### Instruction:'
RESPONSE_KEY = '### Response:'
INPUT_KEY = '### Input:'
HUMAN_ROLE = '[|Human|]'
AI_ROLE = '[|AI|]'


class JaisConfig(openllm_core.LLMConfig):
  """
    Jais-30b-chat-v3 is Jais-30b-v3 fine-tuned over a curated Arabic and English prompt-response pairs dataset. The model architecture is similar to our previous models, Jais-13b-chat, which is based on transformer-based decoder-only (GPT-3) architecture and uses SwiGLU non-linearity. It implements ALiBi position embeddings, enabling the model to extrapolate to long sequence lengths, providing improved context handling and model precision.

  In this release, we have enhanced the model's ability to handle long contexts. The current version can now process up to 8000 tokens, a significant improvement from the 2000-token limit of our previous model.
  """

  __config__ = {
    'name_type': 'lowercase',
    'timeout': 3600000,
    'trust_remote_code': True,
    'url': 'https://huggingface.co/core42',
    'architecture': 'modeling_jais.JAISLMHeadModel',
    'default_id': 'core42/jais-30b-chat-v3',
    'model_ids': ['core42/jais-30b-chat-v3'],
  }

  @property
  def template(self):
    return '{system_message}\n{input_key} {human_role} {instruction}\n{response_key}: {assistant_role}'.format(
      instruction='{instruction}',
      response_key=RESPONSE_KEY,
      system_message='{system_message}',
      human_role=HUMAN_ROLE,
      assistant_role=AI_ROLE,
      input_key=INPUT_KEY,
    )

  @property
  def system_template(self):
    return (
      '### Instruction: Your name is Jais. You can answer in Arabic and English only. '
      'You are a helpful, respectful and honest assistant.\n\n'
      'Complete the conversation below between [|Human|] and [|AI|]:\n'
    )

  class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    do_sample: bool = True

  @property
  def chat_template(self) -> str:
    return repr(
      """{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '###Input: [|Human|] ' + message['content']\n + ' ###Response: [|AI|]' }}{% elif message['role'] == 'system' %}{{ '### Instruction: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"""
    )
