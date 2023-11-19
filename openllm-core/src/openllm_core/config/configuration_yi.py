from __future__ import annotations
import typing as t

import openllm_core
from openllm_core.prompts import PromptTemplate

DEFAULT_SYSTEM_MESSAGE = ''
DEFAULT_PROMPT_TEMPLATE = '{instruction}'


class YiConfig(openllm_core.LLMConfig):
  """The Yi series models are large language models trained from scratch by developers at 01.AI.

  The first public release contains two bilingual(English/Chinese) base models with the parameter sizes of 6B(Yi-6B) and 34B(Yi-34B).
  Both of them are trained with 4K sequence length and can be extended to 32K during inference time. The Yi-6B-200K and Yi-34B-200K are base model with 200K context length.

  See [Yi's GitHub](https://github.com/01-ai/Yi) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://01.ai/',
    'architecture': 'YiForCausalLM',
    'trust_remote_code': True,
    'backend': ('pt', 'vllm'),
    'default_id': '01-ai/Yi-6B',
    'serialisation': 'safetensors',
    'model_ids': ['01-ai/Yi-6B', '01-ai/Yi-34B', '01-ai/Yi-6B-200K', '01-ai/Yi-34B-200K'],
  }

  class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 5
    top_p: float = 0.9
    top_k: int = 40

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5

  @property
  def default_prompt_template(self) -> str:
    return DEFAULT_PROMPT_TEMPLATE

  @property
  def default_system_message(self) -> str:
    return DEFAULT_SYSTEM_MESSAGE

  def sanitize_parameters(
    self,
    prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
    max_new_tokens: int,
    repetition_penalty: float,
    prompt_template: PromptTemplate | str | None = None,
    system_message: str | None = None,
    **attrs: t.Any,
  ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    system_message = DEFAULT_SYSTEM_MESSAGE if system_message is None else system_message
    if prompt_template is None:
      prompt_template = DEFAULT_PROMPT_TEMPLATE
    elif isinstance(prompt_template, str):
      prompt_template = PromptTemplate(template=prompt_template)

    return (
      prompt_template.with_options(system_message=system_message).format(instruction=prompt),
      {'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k},
      {},
    )
