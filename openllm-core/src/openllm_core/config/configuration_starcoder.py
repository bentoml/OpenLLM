from __future__ import annotations
import typing as t

import openllm_core

START_STARCODER_COMMAND_DOCSTRING = '''\
Run a LLMServer for StarCoder model.

\b
> See more information about StarCoder at [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

\b
## Usage

Currently, StarCoder only supports PyTorch. Make sure ``torch`` is available in your system.

\b
StarCoder Runner will use bigcode/starcoder as the default model. To change to any other StarCoder
saved pretrained, or a fine-tune StarCoder, provide ``OPENLLM_MODEL_ID='bigcode/starcoder'``
or provide `--model-id` flag when running ``openllm start starcoder``:

\b
$ openllm start starcoder --model-id 'bigcode/starcoder'
'''
DEFAULT_PROMPT_TEMPLATE = '''{instruction}'''
FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD, EOD, FIM_INDICATOR = '<fim-prefix>', '<fim-middle>', '<fim-suffix>', '<fim-pad>', '<|endoftext|>', '<FILL_HERE>'

class StarCoderConfig(openllm_core.LLMConfig):
  """The StarCoder models are 15.5B parameter models trained on 80+ programming languages from [The Stack (v1.2)](https://huggingface.co/datasets/bigcode/the-stack), with opt-out requests excluded.

  The model uses [Multi Query Attention](https://arxiv.org/abs/1911.02150),
  [a context window of 8192 tokens](https://arxiv.org/abs/2205.14135), and was trained using the
  [Fill-in-the-Middle](https://arxiv.org/abs/2207.14255) objective on 1 trillion tokens.

  Refer to [StarCoder's model card](https://huggingface.co/bigcode/starcoder) for more information.
  """
  __config__ = {
      'name_type': 'lowercase',
      'url': 'https://github.com/bigcode-project/starcoder',
      'architecture': 'GPTBigCodeForCausalLM',
      'requirements': ['bitsandbytes'],
      'default_id': 'bigcode/starcoder',
      'model_ids': ['bigcode/starcoder', 'bigcode/starcoderbase']
  }

  class GenerationConfig:
    temperature: float = 0.2
    max_new_tokens: int = 256
    min_new_tokens: int = 32
    top_k: float = 50
    top_p: float = 0.95
    pad_token_id: int = 49152
    repetition_penalty: float = 1.2

  def sanitize_parameters(self,
                          prompt: str,
                          temperature: float | None = None,
                          top_p: float | None = None,
                          max_new_tokens: int | None = None,
                          repetition_penalty: float | None = None,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    fim_mode, prefix, suffix = FIM_INDICATOR in prompt, None, None
    if fim_mode:
      try:
        prefix, suffix = prompt.split(FIM_INDICATOR)
      except Exception as err:
        raise ValueError(f'Only one {FIM_INDICATOR} allowed in prompt') from err
      prompt_text = f'{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}'
    else:
      prompt_text = prompt
    # XXX: This value for pad_token_id is currently a hack, need more investigate why the default starcoder doesn't include the same value as santacoder EOD
    return prompt_text, {'temperature': temperature, 'top_p': top_p, 'max_new_tokens': max_new_tokens, 'repetition_penalty': repetition_penalty, 'pad_token_id': 49152, **attrs}, {}

  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
    return generation_result[0]
