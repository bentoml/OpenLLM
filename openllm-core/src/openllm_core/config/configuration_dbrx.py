from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class DbrxConfig(openllm_core.LLMConfig):
  """DBRX is a mixture-of-experts (MoE) large language model trained from scratch by Databricks.

  Refer to [Databricks's DBRX page](https://huggingface.co/collections/databricks/dbrx-6601c0852a0cdd3c59f71962) for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'timeout': 3600000,
      'trust_remote_code': True,
      'url': 'https://huggingface.co/collections/databricks/dbrx-6601c0852a0cdd3c59f71962',
      'architecture': 'DbrxForCausalLM',
      'default_id': 'databricks/dbrx-instruct',
      'model_ids': ['databricks/dbrx-instruct', 'databricks/dbrx-base'],
    },
    repr=False,
    exclude=True,
  )

  # NOTE: from get_special_token_id(self.tokenizer, END_KEY)
  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(temperature=0.9, top_p=0.92, top_k=5, max_new_tokens=256)
  )
