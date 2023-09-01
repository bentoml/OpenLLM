from __future__ import annotations
import typing as t

import openllm_core

from openllm_core._prompt import process_prompt

START_BAICHUAN_COMMAND_DOCSTRING = '''\
Run a LLMServer for Baichuan model.

\b
> See more information about Baichuan at [baichuan-inc/Baichuan-7B](https://github.com/baichuan-inc/Baichuan-7B)

\b
## Usage

Currently, Baichuan only supports PyTorch. Make sure ``torch`` is available in your system.

\b
Baichuan Runner will use baichuan-inc/Baichuan-7B as the default model. To change to any other
saved pretrained Baichuan, provide ``OPENLLM_Baichuan_MODEL_ID='fireballoon/baichuan-vicuna-chinese-7b'``
or provide `--model-id` flag when running ``openllm start baichuan``:

\b
$ openllm start baichuan --model-id='fireballoon/baichuan-vicuna-chinese-7b'
'''
DEFAULT_PROMPT_TEMPLATE = '''{instruction}'''

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
      'requires_gpu': True,
      'url': 'https://github.com/baichuan-inc/Baichuan-7B',
      'requirements': ['cpm-kernels', 'sentencepiece'],
      'architecture': 'BaiChuanForCausalLM',
      'default_id': 'baichuan-inc/baichuan-7b',
      'model_ids': [
          'baichuan-inc/baichuan-7b',
          'baichuan-inc/baichuan-13b-base',
          'baichuan-inc/baichuan-13b-chat',
          'fireballoon/baichuan-vicuna-chinese-7b',
          'fireballoon/baichuan-vicuna-7b',
          'hiyouga/baichuan-7b-sft'
      ]
  }

  class GenerationConfig:
    max_new_tokens: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95

  def sanitize_parameters(self,
                          prompt: str,
                          max_new_tokens: int | None = None,
                          top_p: float | None = None,
                          temperature: float | None = None,
                          use_default_prompt_template: bool = False,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {'max_new_tokens': max_new_tokens, 'top_p': top_p, 'temperature': temperature, **attrs}, {}

  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
    return generation_result[0]
