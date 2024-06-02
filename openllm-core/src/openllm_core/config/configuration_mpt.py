from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class MPTConfig(openllm_core.LLMConfig):
  """MPT is a decoder-style transformer pretrained from scratch on English text and code.

  This model was trained by [MosaicML](https://www.mosaicml.com/).

  Refers [HuggingFace's MosaicML page](https://huggingface.co/mosaicml) for more details on specific models.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
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
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(max_new_tokens=128, temperature=0, top_p=0.8)
  )
