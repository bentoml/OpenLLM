from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


docs = """\
OPT was first introduced in [Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) and first released in [metaseq's repository](https://github.com/facebookresearch/metaseq) on May 3rd 2022 by Meta AI.

OPT was predominantly pretrained with English text, but a small amount of non-English data is still present
within the training corpus via CommonCrawl. The model was pretrained using a causal language modeling (CLM)
objective. OPT belongs to the same family of decoder-only models like GPT-3. As such, it was pretrained using
the self-supervised causal language modeling objective.

Refer to [OPT's HuggingFace page](https://huggingface.co/docs/transformers/model_doc/opt) for more information.
"""

metadata_config: ModelSettings = {
  'trust_remote_code': False,
  'url': 'https://huggingface.co/docs/transformers/model_doc/opt',
  'default_id': 'facebook/opt-1.3b',
  'architecture': 'OPTForCausalLM',
  'model_ids': [
    'facebook/opt-125m',
    'facebook/opt-350m',
    'facebook/opt-1.3b',
    'facebook/opt-2.7b',
    'facebook/opt-6.7b',
    'facebook/opt-66b',
  ],
  'fine_tune_strategies': (
    {
      'adapter_type': 'lora',
      'r': 16,
      'lora_alpha': 32,
      'target_modules': ['q_proj', 'v_proj'],
      'lora_dropout': 0.05,
      'bias': 'none',
    },
  ),
}

generation_config: openllm_core.GenerationConfig = openllm_core.GenerationConfig.model_construct(
  top_k=15, temperature=0.75, max_new_tokens=256, num_return_sequences=1
)

OPTConfig = pydantic.create_model(
  'OPTConfig',
  __doc__=docs,
  __base__=(openllm_core.LLMConfig,),
  metadata_config=(ModelSettings, pydantic.Field(default=metadata_config, repr=False, exclude=True)),
  generation_config=(openllm_core.GenerationConfig, pydantic.Field(default=generation_config)),
)
