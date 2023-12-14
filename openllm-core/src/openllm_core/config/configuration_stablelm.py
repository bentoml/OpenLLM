from __future__ import annotations

import openllm_core


class StableLMConfig(openllm_core.LLMConfig):
  """StableLM-Base-Alpha is a suite of 3B and 7B parameter decoder-only language models.

  It is pre-trained on a diverse collection of English datasets with a sequence
  length of 4096 to push beyond the context window limitations of existing open-source language models.

  StableLM-Tuned-Alpha is a suite of 3B and 7B parameter decoder-only language models
  built on top of the StableLM-Base-Alpha models and further fine-tuned on various chat and
  instruction-following datasets.

  Refer to [StableLM-tuned's model card](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
  and [StableLM-base's model card](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)
  for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://github.com/Stability-AI/StableLM',
    'architecture': 'GPTNeoXForCausalLM',
    'default_id': 'stabilityai/stablelm-tuned-alpha-3b',
    'model_ids': [
      'stabilityai/stablelm-tuned-alpha-3b',
      'stabilityai/stablelm-tuned-alpha-7b',
      'stabilityai/stablelm-base-alpha-3b',
      'stabilityai/stablelm-base-alpha-7b',
    ],
  }

  class GenerationConfig:
    temperature: float = 0.9
    max_new_tokens: int = 128
    top_k: int = 0
    top_p: float = 0.9

  @property
  def template(self) -> str:
    return '{system_message}<|USER|>{instruction}<|ASSISTANT|>'

  @property
  def system_message(self) -> str:
    return """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
