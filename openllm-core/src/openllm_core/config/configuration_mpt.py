from __future__ import annotations
import typing as t

import openllm_core

from openllm_core._prompt import process_prompt
from openllm_core.utils import dantic

MPTPromptType = t.Literal['default', 'instruct', 'chat', 'storywriter']

START_MPT_COMMAND_DOCSTRING = '''\
Run a LLMServer for MPT model.

\b
> See more information about MPT at [HuggingFace's MosaicML page](https://huggingface.co/mosaicml)

\b
## Usage

Currently, MPT only supports PyTorch. Make sure ``torch`` is available in your system.

If you want to use Flash Attention support with openai/triton, make sure to install OpenLLM with

\b
```bash
pip install "openllm[mpt]"
```

\b
MPT Runner will use mosaicml/mpt-7b-instruct as the default model. To change to any other MPT
saved pretrained, or a fine-tune MPT, provide ``OPENLLM_MPT_MODEL_ID='mosaicml/mpt-30b'``
or provide `--model-id` flag when running ``openllm start mpt``:

\b
$ openllm start mpt --model-id mosaicml/mpt-30b
'''
INSTRUCTION_KEY, RESPONSE_KEY, END_KEY = '### Instruction:', '### Response:', '### End'
INTRO_BLURB = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
# NOTE: This is the prompt that is used for generating responses using an already
# trained model.  It ends with the response key, where the job of the model is to provide
# the completion that follows it (i.e. the response itself).
_chat_prompt, _default_prompt, _instruct_prompt = '''{instruction}''', '''{instruction}''', '''{intro}
{instruction_key}
{instruction}
{response_key}
'''.format(intro=INTRO_BLURB, instruction_key=INSTRUCTION_KEY, instruction='{instruction}', response_key=RESPONSE_KEY)
PROMPT_MAPPING = {'default': _default_prompt, 'instruct': _instruct_prompt, 'storywriter': _default_prompt, 'chat': _chat_prompt}

def _get_prompt(model_type: str) -> str:
  return PROMPT_MAPPING[model_type]

DEFAULT_PROMPT_TEMPLATE = _get_prompt

class MPTConfig(openllm_core.LLMConfig):
  """MPT is a decoder-style transformer pretrained from scratch on English text and code.

  This model was trained by [MosaicML](https://www.mosaicml.com/).

  ``openllm.MPT`` encapsulate a family of MPT variants that is publicly available
  on HuggingFace. Refers [HuggingFace's MosaicML page](https://huggingface.co/mosaicml)
  for more details on specific models.
  """
  __config__ = {
      'name_type': 'lowercase',
      'trust_remote_code': True,
      'url': 'https://huggingface.co/mosaicml',
      'timeout': int(36e6),
      'requirements': ['triton', 'einops'],
      'architecture': 'MPTForCausalLM',
      'default_id': 'mosaicml/mpt-7b-instruct',
      'model_ids': [
          'mosaicml/mpt-7b', 'mosaicml/mpt-7b-instruct', 'mosaicml/mpt-7b-chat', 'mosaicml/mpt-7b-storywriter', 'mosaicml/mpt-30b', 'mosaicml/mpt-30b-instruct', 'mosaicml/mpt-30b-chat'
      ]
  }
  prompt_type: MPTPromptType = dantic.Field('"default"', description='Given prompt type for running MPT. Default will be inferred from model name if pretrained.')
  max_sequence_length: int = dantic.Field(
      2048,
      description=
      'Max sequence length to run MPT with. Note that MPT is trained ith sequence length of 2048, but with [ALiBi](https://arxiv.org/abs/2108.12409) it can set up to 4096 (for 7b models) and 16384 (for 30b models)'
  )

  class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0
    top_p: float = 0.8

  def sanitize_parameters(self,
                          prompt: str,
                          max_new_tokens: int | None = None,
                          temperature: float | None = None,
                          top_p: float | None = None,
                          prompt_type: MPTPromptType | None = None,
                          use_default_prompt_template: bool = True,
                          **attrs: t.Any,
                          ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    _template = None
    if use_default_prompt_template:
      if prompt_type is None:
        if 'instruct' in self.model_id: prompt_type = 'instruct'
        elif 'storywriter' in self.model_id: prompt_type = 'storywriter'
        elif 'chat' in self.model_id: prompt_type = 'chat'
        else: prompt_type = 'default'
      _template = DEFAULT_PROMPT_TEMPLATE(prompt_type)
    return process_prompt(prompt, _template, use_default_prompt_template), {'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p}, {}

  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **attrs: t.Any) -> str:
    return generation_result[0]
