from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class GPTNeoXConfig(openllm_core.LLMConfig):
  """GPTNeoX is an autoregressive language model trained on the Pile, whose weights will be made freely and openly available to the public through a permissive license.

  It is, to the best of our knowledge, the largest dense autoregressive model
  that has publicly available weights at the time of submission. The training and evaluation code, as well as the model weights,
  can be found at https://github.com/EleutherAI/gpt-neox.

  GPTNeoX has been used to fine-tune on various models, such as Dolly, StableLM, and Pythia.

  Note that OpenLLM provides first-class support for all of the aforementioned model. Users can
  also use `openllm start gpt-neox` to run all of the GPTNeoX variant's model

  Refer to [GPTNeoX's model card](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
  for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'architecture': 'GPTNeoXForCausalLM',
      # NOTE: See https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B
      'url': 'https://github.com/EleutherAI/gpt-neox',
      'default_id': 'eleutherai/gpt-neox-20b',
      'model_ids': ['eleutherai/gpt-neox-20b'],
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(temperature=0.9, max_new_tokens=100)
  )
