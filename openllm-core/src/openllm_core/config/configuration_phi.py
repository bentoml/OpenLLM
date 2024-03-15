from __future__ import annotations

import openllm_core, typing as t

if t.TYPE_CHECKING:
  from openllm_core._schemas import MessageParam


class PhiConfig(openllm_core.LLMConfig):
  """The language model phi-1.5 is a Transformer with 1.3 billion parameters.

  It was trained using the same data sources as [phi-1](https://huggingface.co/microsoft/phi-1), augmented with a new data source that consists of various
  NLP synthetic texts. When assessed against benchmarks testing common sense,
  language understanding, and logical reasoning, phi-1.5 demonstrates a nearly state-of-the-art performance among models with less
  than 10 billion parameters.

  Refer to [Phi's HuggingFace repos](https://huggingface.co/microsoft/phi-1_5)
  for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://arxiv.org/abs/2309.05463',
    'architecture': 'PhiForCausalLM',
    'trust_remote_code': True,
    'backend': ('pt', 'vllm'),
    'default_id': 'microsoft/phi-1_5',
    'serialisation': 'safetensors',
    'model_ids': ['microsoft/phi-1_5'],
    'fine_tune_strategies': ({'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'},),
  }

  class GenerationConfig:
    max_new_tokens: int = 200

  class SamplingParams:
    best_of: int = 1

  @property
  def chat_template(self) -> str:
    return repr(
      "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )

  @property
  def chat_messages(self) -> list[MessageParam]:
    from openllm_core._schemas import MessageParam

    return [
      MessageParam(role='user', content="I don't know why, I'm struggling to maintain focus while studying. Any suggestions?"),
      MessageParam(role='assistant', content='Have you tried using a timer? It can help you stay on track and avoid distractions.'),
      MessageParam(role='user', content="That's a good idea. I'll give it a try. What else can I do to boost my productivity?"),
    ]
