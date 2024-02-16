from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


docs = """\
The Yi series models are large language models trained from scratch by developers at 01.AI.

The first public release contains two bilingual(English/Chinese) base models with the parameter sizes of 6B(Yi-6B) and 34B(Yi-34B).
Both of them are trained with 4K sequence length and can be extended to 32K during inference time. The Yi-6B-200K and Yi-34B-200K are base model with 200K context length.

See [Yi's GitHub](https://github.com/01-ai/Yi) for more information.
"""

metadata_config: ModelSettings = {
  'name_type': 'lowercase',
  'url': 'https://01.ai/',
  'architecture': 'YiForCausalLM',
  'trust_remote_code': True,
  'backend': ('pt', 'vllm'),
  'default_id': '01-ai/Yi-6B',
  'serialisation': 'safetensors',
  'model_ids': ['01-ai/Yi-6B', '01-ai/Yi-34B', '01-ai/Yi-6B-200K', '01-ai/Yi-34B-200K'],
}

generation_config: openllm_core.GenerationConfig = openllm_core.GenerationConfig(
  max_new_tokens=256,
  temperature=0.7,
  repetition_penalty=1.3,
  no_repeat_ngram_size=5,
  top_p=0.9,
  top_k=40,
  best_of=1,
  presence_penalty=0.5,
)

YiConfig = pydantic.create_model(
  'YiConfig',
  __doc__=docs,
  __base__=(openllm_core.LLMConfig,),
  metadata_config=(ModelSettings, pydantic.Field(default=metadata_config, repr=False, exclude=True)),
  generation_config=(openllm_core.GenerationConfig, pydantic.Field(default=generation_config)),
)
