from __future__ import annotations
import typing as t

import openllm_core

from openllm_core._prompt import process_prompt
from openllm_core.utils import dantic

START_OPT_COMMAND_DOCSTRING = '''\
Run a LLMServer for OPT model.

\b
> See more information about falcon at [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)

\b
## Usage

By default, this model will use the PyTorch model for inference. However, this model supports both Flax and Tensorflow.

\b
- To use Flax, set the environment variable ``OPENLLM_BACKEND="flax"``

\b
- To use Tensorflow, set the environment variable ``OPENLLM_BACKEND="tf"``

\b
OPT Runner will use facebook/opt-2.7b as the default model. To change to any other OPT
saved pretrained, or a fine-tune OPT, provide ``OPENLLM_MODEL_ID='facebook/opt-6.7b'``
or provide `--model-id` flag when running ``openllm start opt``:

\b
$ openllm start opt --model-id facebook/opt-6.7b
'''
DEFAULT_PROMPT_TEMPLATE = '''{instruction}'''

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
      'fine_tune_strategies': ({
          'adapter_type': 'lora', 'r': 16, 'lora_alpha': 32, 'target_modules': ['q_proj', 'v_proj'], 'lora_dropout': 0.05, 'bias': 'none'
      },)
  }
  format_outputs: bool = dantic.Field(False, description='''Whether to format the outputs. This can be used when num_return_sequences > 1.''')

  class GenerationConfig:
    top_k: int = 15
    temperature: float = 0.75
    max_new_tokens: int = 1024
    num_return_sequences: int = 1

  def sanitize_parameters(self,
                          prompt: str,
                          max_new_tokens: int | None = None,
                          temperature: float | None = None,
                          top_k: int | None = None,
                          num_return_sequences: int | None = None,
                          use_default_prompt_template: bool = False,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {
        'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_k': top_k, 'num_return_sequences': num_return_sequences
    }, {}

  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **attrs: t.Any) -> str:
    if len(generation_result) == 1: return generation_result[0]
    if self.config.format_outputs: return 'Generated result:\n' + '\n -'.join(generation_result)
    else: return '\n'.join(generation_result)
