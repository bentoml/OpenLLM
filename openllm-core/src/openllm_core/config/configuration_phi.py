from __future__ import annotations
import typing as t

import openllm_core
from openllm_core.prompts import PromptTemplate

DEFAULT_PROMPT_TEMPLATE = """{instruction}"""

DEFAULT_SYSTEM_MESSAGE = ''


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
    'fine_tune_strategies': (
      {'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'},
    ),
  }

  class GenerationConfig:
    max_new_tokens: int = 200

  class SamplingParams:
    best_of: int = 1

  @property
  def default_prompt_template(self) -> str:
    return DEFAULT_PROMPT_TEMPLATE

  @property
  def default_system_message(self) -> str:
    return DEFAULT_SYSTEM_MESSAGE

  def sanitize_parameters(
    self,
    prompt: str,
    prompt_template: PromptTemplate | str | None = None,
    system_message: str | None = None,
    max_new_tokens: int | None = None,
    **attrs: t.Any,
  ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    system_message = DEFAULT_SYSTEM_MESSAGE if system_message is None else system_message
    if prompt_template is None:
      prompt_template = PromptTemplate(template=self.default_prompt_template)
    elif isinstance(prompt_template, str):
      prompt_template = PromptTemplate(template=prompt_template)
    return (
      prompt_template.with_options(system_message=system_message).format(instruction=prompt),
      {'max_new_tokens': max_new_tokens},
      {},
    )
