from __future__ import annotations

import openllm_core


class OPTConfig(openllm_core.LLMConfig):
  """OPT was first introduced in [Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) and first released in [metaseq's repository](https://github.com/facebookresearch/metaseq) on May 3rd 2022 by Meta AI.

  OPT was predominantly pretrained with English text, but a small amount of non-English data is still present
  within the training corpus via CommonCrawl. The model was pretrained using a causal language modeling (CLM)
  objective. OPT belongs to the same family of decoder-only models like GPT-3. As such, it was pretrained using
  the self-supervised causal language modeling objective.

  Refer to [OPT's HuggingFace page](https://huggingface.co/docs/transformers/model_doc/opt) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'trust_remote_code': False,
    'url': 'https://huggingface.co/docs/transformers/model_doc/opt',
    'default_id': 'facebook/opt-1.3b',
    'architecture': 'OPTForCausalLM',
    'model_ids': ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-66b'],
    'fine_tune_strategies': (
      {'adapter_type': 'lora', 'r': 16, 'lora_alpha': 32, 'target_modules': ['q_proj', 'v_proj'], 'lora_dropout': 0.05, 'bias': 'none'},
    ),
  }

  class GenerationConfig:
    top_k: int = 15
    temperature: float = 0.75
    max_new_tokens: int = 256
    num_return_sequences: int = 1
