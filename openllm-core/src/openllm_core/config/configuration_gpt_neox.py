from __future__ import annotations
import typing as t

import openllm_core

from openllm_core._prompt import process_prompt
from openllm_core.utils import dantic

START_GPT_NEOX_COMMAND_DOCSTRING = '''\
Run a LLMServer for GPTNeoX model.

\b
> See more information about GPTNeoX at [HuggingFace's model card](https://huggingface.co/docs/transformers/model_doc/gpt_neox)

\b
## Usage

Currently, GPTNeoX only supports PyTorch. Make sure ``torch`` is available in your system.

\b
GPTNeoX Runner will use EleutherAI/gpt-neox-20b as the default model. To change to any other GPTNeoX
saved pretrained, or a fine-tune GPTNeoX, provide ``OPENLLM_GPT_NEOX_MODEL_ID='stabilityai/stablelm-tuned-alpha-3b'``
or provide `--model-id` flag when running ``openllm start gpt-neox``:

\b
$ openllm start gpt-neox --model-id 'stabilityai/stablelm-tuned-alpha-3b'
'''
DEFAULT_PROMPT_TEMPLATE = '''{instruction}'''

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
  __config__ = {
      'model_name': 'gpt_neox',
      'start_name': 'gpt-neox',
      'requires_gpu': True,
      'architecture': 'GPTNeoXForCausalLM',
      'url': 'https://github.com/EleutherAI/gpt-neox',
      'default_id': 'eleutherai/gpt-neox-20b',
      'model_ids': ['eleutherai/gpt-neox-20b']
  }
  use_half_precision: bool = dantic.Field(True, description='Whether to use half precision for model.')

  class GenerationConfig:
    temperature: float = 0.9
    max_new_tokens: int = 100

  def sanitize_parameters(self,
                          prompt: str,
                          temperature: float | None = None,
                          max_new_tokens: int | None = None,
                          use_default_prompt_template: bool = True,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {'max_new_tokens': max_new_tokens, 'temperature': temperature}, {}

  def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str:
    return generation_result[0]
