from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class StarCoderConfig(openllm_core.LLMConfig):
  """The StarCoder models are 15.5B parameter models trained on 80+ programming languages from [The Stack (v1.2)](https://huggingface.co/datasets/bigcode/the-stack), with opt-out requests excluded.

  The model uses [Multi Query Attention](https://arxiv.org/abs/1911.02150),
  [a context window of 8192 tokens](https://arxiv.org/abs/2205.14135), and was trained using the
  [Fill-in-the-Middle](https://arxiv.org/abs/2207.14255) objective on 1 trillion tokens.

  Refer to [StarCoder's model card](https://huggingface.co/bigcode/starcoder) for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'url': 'https://github.com/bigcode-project/starcoder',
      'architecture': 'GPTBigCodeForCausalLM',
      'requirements': ['bitsandbytes'],
      'default_id': 'bigcode/starcoder',
      'model_ids': ['bigcode/starcoder', 'bigcode/starcoderbase'],
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(
      temperature=0.2,
      max_new_tokens=256,
      min_new_tokens=32,
      top_k=50,
      top_p=0.95,
      pad_token_id=49152,
      repetition_penalty=1.2,
    )
  )
