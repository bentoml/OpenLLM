from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class CohereConfig(openllm_core.LLMConfig):
  """C4AI Command R+ is an open weights research release of a 104B billion parameter model with highly
  advanced capabilities, this includes Retrieval Augmented Generation (RAG) and tool use to
  automate sophisticated tasks.

  Refer to [CohereForAI's org card](https://huggingface.co/CohereForAI) for more information on Command-R
  for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'name_type': 'lowercase',
      'url': 'https://huggingface.co/CohereForAI',
      'architecture': 'CohereForCausalLM',
      'default_id': 'CohereForAI/c4ai-command-r-plus',
      'serialisation': 'safetensors',
      'model_ids': ['CohereForAI/c4ai-command-r-plus', 'CohereForAI/c4ai-command-r-v01'],
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(
      max_new_tokens=128, temperature=0.6, top_p=0.9, top_k=12, best_of=1, presence_penalty=0.5
    )
  )
