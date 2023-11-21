from __future__ import annotations

import openllm_core


class ChatGLMConfig(openllm_core.LLMConfig):
  """ChatGLM is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM) framework.

  With the quantization technique, users can deploy locally on consumer-grade graphics cards
  (only 6GB of GPU memory is required at the INT4 quantization level).

  ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue.
  The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning,
  feedback bootstrap, and reinforcement learning wit human feedback.
  With only about 6.2 billion parameters, the model is able to generate answers that are in line
  with human preference.

  Refer to [ChatGLM's GitHub page](https://github.com/THUDM/ChatGLM-6B) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'trust_remote_code': True,
    'timeout': 3600000,
    'url': 'https://github.com/THUDM/ChatGLM-6B',
    'requirements': ['cpm-kernels'],
    'backend': ('pt', 'vllm'),
    'architecture': 'ChatGLMModel',
    'default_id': 'thudm/chatglm-6b',
    'model_ids': [
      'thudm/chatglm-6b',
      'thudm/chatglm-6b-int8',
      'thudm/chatglm-6b-int4',
      'thudm/chatglm2-6b',
      'thudm/chatglm2-6b-int4',
      'thudm/chatglm3-6b',
    ],
  }

  class GenerationConfig:
    max_new_tokens: int = 2048
    num_beams: int = 1
    top_p: float = 0.7
    temperature: float = 0.95
