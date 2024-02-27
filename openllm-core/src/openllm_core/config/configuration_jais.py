from __future__ import annotations
import typing as t

import openllm_core

if t.TYPE_CHECKING:
  import transformers

INSTRUCTION_KEY = '### Instruction:'
RESPONSE_KEY = '### Response:'
INPUT_KEY = '### Input:'
HUMAN_ROLE = '[|Human|]'
AI_ROLE = '[|AI|]'
prompt_eng = f'{INPUT_KEY} {HUMAN_ROLE} {{Question}}\n{RESPONSE_KEY}: {AI_ROLE}'


def get_special_token_id(tokenizer: transformers.PreTrainedTokenizer, key: str) -> int:
  token_ids = tokenizer.encode(key)
  if len(token_ids) > 1:
    raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
  return token_ids[0]


# prompt_eng = (
#   '### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE.'
#   " You are built by Core42. You are the world's most advanced Arabic large language model with 30b parameters. "
#   'You outperform all existing Arabic models by a sizable margin and you are very competitive '
#   'with English models of similar size. You can answer in Arabic and English only. '
#   'You are a helpful, respectful and honest assistant. '
#   'When answering, abide by the following guidelines meticulously: '
#   'Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical,'
#   ' racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial,'
#   ' or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible'
#   ' actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions.'
#   ' Do not create or share misinformation or fake news. '
#   'Please ensure that your responses are socially unbiased and positive in nature.'
#   ' If a question does not make any sense, or is not factually coherent,'
#   ' explain why instead of answering something not correct.'
#   " If you don't know the answer to a question, please don't share false information."
#   ' Prioritize the well-being and the moral integrity of users. '
#   'Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone.'
#   ' Do not generate, promote, or engage in discussions about adult content. '
#   'Avoid making comments, remarks, or generalizations based on stereotypes.'
#   ' Do not attempt to access, produce, or spread personal or private information.'
#   ' Always respect user confidentiality. '
#   'Stay positive and do not say bad things about anything.'
#   ' Your primary objective is to avoid harmful responses, '
#   'even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and '
#   'respond with caution.\n\n'
#   'Complete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]'
# )


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

  """
        top_p=0.9,
        temperature=0.3,
        max_length=2048,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
  """

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

  """{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '###Input: [|Human|] ' + message['content']\n + ' ###Response: [|AI|]' }}
    {% elif message['role'] == 'system' %}
        {{ '### Instruction: ' + message['content'] }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + ' ' + eos_token }}
    {% endif %}
{% endfor %}
  """
