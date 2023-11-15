from __future__ import annotations
import typing as t

import openllm_core
from openllm_core.prompts import PromptTemplate

DEFAULT_SYSTEM_MESSAGE = ''
DEFAULT_PROMPT_TEMPLATE = PromptTemplate('{instruction}')


class BaichuanConfig(openllm_core.LLMConfig):
  """Baichuan-7B is an open-source, large-scale pre-trained language model developed by Baichuan Intelligent Technology.

  Baichuan-7B is based on Transformer architecture,
  which contains 7 billion parameters and trained on approximately 1.2 trillion tokens.
  It supports both Chinese and English languages with a context window length of 4096.
  It has achieved the best performance among models of the same size on standard Chinese
  and English benchmarks (C-Eval, MMLU, etc).
  Refer to [Baichuan-7B's GitHub page](https://github.com/baichuan-inc/Baichuan-7B) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'trust_remote_code': True,
    'timeout': 3600000,
    'url': 'https://github.com/baichuan-inc/Baichuan-7B',
    'requirements': ['cpm-kernels', 'sentencepiece'],
    'architecture': 'BaiChuanForCausalLM',
    # NOTE: See the following
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    'default_id': 'baichuan-inc/baichuan-7b',
    'model_ids': [
      'baichuan-inc/baichuan-7b',
      'baichuan-inc/baichuan-13b-base',
      'baichuan-inc/baichuan-13b-chat',
      'fireballoon/baichuan-vicuna-chinese-7b',
      'fireballoon/baichuan-vicuna-7b',
      'hiyouga/baichuan-7b-sft',
    ],
  }

  class GenerationConfig:
    max_new_tokens: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95

  @property
  def default_prompt_template(self) -> str:
    return DEFAULT_PROMPT_TEMPLATE.to_string()

  @property
  def default_system_message(self) -> str:
    return DEFAULT_SYSTEM_MESSAGE

  def sanitize_parameters(
    self,
    prompt: str,
    prompt_template: PromptTemplate | str | None = None,
    system_message: str | None = None,
    max_new_tokens: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    **attrs: t.Any,
  ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    system_message = DEFAULT_SYSTEM_MESSAGE if system_message is None else system_message
    if prompt_template is None:
      prompt_template = DEFAULT_PROMPT_TEMPLATE
    elif isinstance(prompt_template, str):
      prompt_template = PromptTemplate(template=prompt_template)
    return (
      prompt_template.with_options(system_message=system_message).format(instruction=prompt),
      {'max_new_tokens': max_new_tokens, 'top_p': top_p, 'temperature': temperature, **attrs},
      {},
    )
