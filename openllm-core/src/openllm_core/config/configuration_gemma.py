from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class GemmaConfig(openllm_core.LLMConfig):
  """Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants.

  Refer to [Gemma's model card](https://ai.google.dev/gemma/docs) for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'name_type': 'lowercase',
      'url': 'https://ai.google.dev/gemma/docs',
      'architecture': 'GemmaForCausalLM',
      'default_id': 'google/gemma-7b',
      'serialisation': 'safetensors',
      'model_ids': ['google/gemma-7b', 'google/gemma-7b-it', 'google/gemma-2b', 'google/gemma-2b-it'],
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(
      max_new_tokens=128, temperature=0.6, top_p=0.9, top_k=12, best_of=1, presence_penalty=0.5
    )
  )
