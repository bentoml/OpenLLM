from __future__ import annotations

import openllm_core


class QwenConfig(openllm_core.LLMConfig):
  """Qwen-7B is the 7B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen),
  proposed by Alibaba Cloud. Qwen-14B is a Transformer-based large language model,
  which is pretrained on a large volume of data, including web texts, books, codes, etc.
  Additionally, based on the pretrained Qwen-14B, we release Qwen-14B-Chat, a large-model-based AI assistant,
  which is trained with alignment techniques.
  Refer to [Qwen's GitHub page](https://github.com/QwenLM/Qwen) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'trust_remote_code': True,
    'timeout': 3600000,
    'url': 'https://github.com/QwenLM/Qwen',
    'requirements': ['cpm-kernels', 'tiktoken'],
    'backend': ('pt', 'vllm'),
    'architecture': 'QWenLMHeadModel',
    'default_id': 'qwen/Qwen-7B-Chat',
    'model_ids': [
      'qwen/Qwen-7B-Chat',
      'qwen/Qwen-7B-Chat-Int8',
      'qwen/Qwen-7B-Chat-Int4',
      'qwen/Qwen-14B-Chat',
      'qwen/Qwen-14B-Chat-Int8',
      'qwen/Qwen-14B-Chat-Int4',
    ],
  }

  class GenerationConfig:
    max_new_tokens: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95
