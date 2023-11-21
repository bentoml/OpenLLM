from __future__ import annotations

import openllm_core


class MPTConfig(openllm_core.LLMConfig):
  """MPT is a decoder-style transformer pretrained from scratch on English text and code.

  This model was trained by [MosaicML](https://www.mosaicml.com/).

  ``openllm.MPT`` encapsulate a family of MPT variants that is publicly available
  on HuggingFace. Refers [HuggingFace's MosaicML page](https://huggingface.co/mosaicml)
  for more details on specific models.
  """

  __config__ = {
    'name_type': 'lowercase',
    'trust_remote_code': True,
    'url': 'https://huggingface.co/mosaicml',
    'timeout': int(36e6),
    'requirements': ['triton'],
    'architecture': 'MPTForCausalLM',
    # NOTE: See https://huggingface.co/TheBloke/mpt-30B-chat-GGML/discussions/4
    'default_id': 'mosaicml/mpt-7b-instruct',
    'model_ids': [
      'mosaicml/mpt-7b',
      'mosaicml/mpt-7b-instruct',
      'mosaicml/mpt-7b-chat',
      'mosaicml/mpt-7b-storywriter',
      'mosaicml/mpt-30b',
      'mosaicml/mpt-30b-instruct',
      'mosaicml/mpt-30b-chat',
    ],
  }

  class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0
    top_p: float = 0.8
