from __future__ import annotations

import openllm_core

START_MISTRAL_COMMAND_DOCSTRING = '''\
Run a LLMServer for Mistral model.

\b
> See more information about Mistral at [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

\b
## Usage

By default, this model will use the PyTorch model for inference. However, if vLLM exists, then it will use vLLM instead.

\b
- To use vLLM, set the environment variable ``OPENLLM_BACKEND="vllm"``

\b
Mistral Runner will use mistralai/Mistral-7B-Instruct-v0.1 as the default model. To change to any other Mistral
saved pretrained, or a fine-tune Mistral, provide ``OPENLLM_MODEL_ID='HuggingFaceH4/zephyr-7b-alpha'``
or provide `--model-id` flag when running ``openllm start mistral``:

\b
$ openllm start mistral --model-id HuggingFaceH4/zephyr-7b-alpha
'''
DEFAULT_PROMPT_TEMPLATE = '''{instruction}'''

class MistralConfig(openllm_core.LLMConfig):
  """Mistral's [paper](https://arxiv.org/abs/2310.06825) and first released by [MistralAI](https://mistral.ai/news/announcing-mistral-7b/).

  Mistral-7B-v0.1 is Mistral AI\'s first Large Language Model (LLM).
  Refer to [Mistral's HuggingFace page](https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/mistral#overview) for more information.
  """
  __config__ = {
      'name_type': 'lowercase',
      'url': 'https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/mistral#overview',
      'default_id': 'mistralai/Mistral-7B-Instruct-v0.1',
      'architecture': 'MistralForCausalLM',
      'add_generation_prompt': True,
      'model_ids': ['mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1', 'amazon/MistralLite', 'HuggingFaceH4/zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-alpha'],
  }

  class GenerationConfig:
    top_k: int = 12
    temperature: float = 0.75
    max_new_tokens: int = 256
