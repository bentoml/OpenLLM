from __future__ import annotations

import openllm_core


class FalconConfig(openllm_core.LLMConfig):
  """Falcon-7B is a 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora.

  It is made available under the TII Falcon LLM License.

  Refer to [Falcon's HuggingFace page](https://huggingface.co/tiiuae/falcon-7b) for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'trust_remote_code': False,
    'timeout': int(36e6),
    'url': 'https://falconllm.tii.ae/',
    'requirements': ['xformers'],
    'architecture': 'FalconForCausalLM',
    # NOTE: See https://huggingface.co/tiiuae/falcon-7b-instruct/discussions/1
    'default_id': 'tiiuae/falcon-7b',
    'model_ids': ['tiiuae/falcon-7b', 'tiiuae/falcon-40b', 'tiiuae/falcon-7b-instruct', 'tiiuae/falcon-40b-instruct'],
    'fine_tune_strategies': (
      {
        'adapter_type': 'lora',
        'r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'bias': 'none',
        'target_modules': ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
      },
    ),
  }

  class GenerationConfig:
    max_new_tokens: int = 200
    top_k: int = 10
    num_return_sequences: int = 1
    num_beams: int = 4

  @property
  def template(self) -> str:
    return '{context}\n{user_name}: {instruction}\n{agent}:'
