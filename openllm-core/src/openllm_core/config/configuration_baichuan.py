from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


docs = """\
Baichuan-7B is an open-source, large-scale pre-trained language model developed by Baichuan Intelligent Technology.

Baichuan-7B is based on Transformer architecture,
which contains 7 billion parameters and trained on approximately 1.2 trillion tokens.
It supports both Chinese and English languages with a context window length of 4096.
It has achieved the best performance among models of the same size on standard Chinese
and English benchmarks (C-Eval, MMLU, etc).
Refer to [Baichuan-7B's GitHub page](https://github.com/baichuan-inc/Baichuan-7B) for more information.
"""

metadata_config: ModelSettings = {
  'trust_remote_code': True,
  'timeout': 3600000,
  'url': 'https://github.com/baichuan-inc/Baichuan-7B',
  'requirements': ['cpm-kernels'],
  'architecture': 'BaichuanForCausalLM',
  # NOTE: See the following
  # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
  # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
  # https://github.com/baichuan-inc/Baichuan-13B/issues/25
  'default_id': 'baichuan-inc/baichuan-7b',
  'model_ids': [
    'baichuan-inc/baichuan2-7b-base',
    'baichuan-inc/baichuan2-7b-chat',
    'baichuan-inc/baichuan2-13b-base',
    'baichuan-inc/baichuan2-13b-chat',
  ],
}

generation_config = openllm_core.GenerationConfig.model_construct(max_new_tokens=2048, top_p=0.7, temperature=0.95)

BaichuanConfig = pydantic.create_model(
  'BaichuanConfig',
  __doc__=docs,
  __base__=(openllm_core.LLMConfig,),
  metadata_config=(ModelSettings, pydantic.Field(default=metadata_config, repr=False, exclude=True)),
  generation_config=(openllm_core.GenerationConfig, pydantic.Field(default=generation_config)),
)
