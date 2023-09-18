from __future__ import annotations
import typing as t

import openllm_core

from openllm_core._prompt import process_prompt
from openllm_core.utils import dantic

START_LLAMA_COMMAND_DOCSTRING = '''\
Run a LLMServer for Llama model.

\b
> See more information about Llama at [Llama's model card](https://huggingface.co/docs/transformers/main/model_doc/llama

\b
## Usage

By default, this model will use [vLLM](https://github.com/vllm-project/vllm) for inference.
This model will also supports PyTorch.

\b
- To use PyTorch, set the environment variable ``OPENLLM_BACKEND="pt"``

\b
- To use vLLM, set the environment variable ``OPENLLM_BACKEND="vllm"``

\b
Llama Runner will use decapoda-research/llama-7b-hf as the default model. To change to any other Llama
saved pretrained, or a fine-tune Llama, provide ``OPENLLM_MODEL_ID='openlm-research/open_llama_7b_v2'``
or provide `--model-id` flag when running ``openllm start llama``:

\b
$ openllm start llama --model-id 'openlm-research/open_llama_7b_v2'

\b
OpenLLM also supports running Llama-2 and its fine-tune and variants. To import the Llama weights, one can use the following:

\b
$ CONVERTER=hf-llama2 openllm import llama /path/to/llama-2
'''
SYSTEM_MESSAGE = '''
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
'''
SINST_KEY, EINST_KEY, SYS_KEY, EOS_TOKEN, BOS_TOKEN = '[INST]', '[/INST]', '<<SYS>>', '</s>', '<s>'
# TODO: support history and v1 prompt implementation
_v1_prompt, _v2_prompt = '''{instruction}''', '''{start_key} {sys_key}\n{system_message}\n{sys_key}\n\n{instruction}\n{end_key} '''.format(start_key=SINST_KEY,
                                                                                                                                           sys_key=SYS_KEY,
                                                                                                                                           system_message=SYSTEM_MESSAGE,
                                                                                                                                           instruction='{instruction}',
                                                                                                                                           end_key=EINST_KEY)
PROMPT_MAPPING = {'v1': _v1_prompt, 'v2': _v2_prompt}

def _get_prompt(model_type: t.Literal['v1', 'v2']) -> str:
  return PROMPT_MAPPING[model_type]

DEFAULT_PROMPT_TEMPLATE = _get_prompt

class LlamaConfig(openllm_core.LLMConfig):
  """LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.

  It is a collection of foundation language models ranging from 7B to 65B parameters.

  Llama also include support for the recent propsed [Llama-2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

  Note that all variants of Llama including fine-tuning, quantisation format are all supported with ``openllm.Llama``.

  Refer to [Llama's model card](https://huggingface.co/docs/transformers/main/model_doc/llama)
  for more information.
  """
  use_llama2_prompt: bool = dantic.Field(False, description='Whether to use the prompt format for Llama 2. Disable this when working with Llama 1.')
  __config__ = {
      'name_type': 'lowercase',
      'url': 'https://github.com/facebookresearch/llama',
      'default_backend': {
          'cpu': 'pt', 'nvidia.com/gpu': 'pt'
      },
      'architecture': 'LlamaForCausalLM',
      'requirements': ['fairscale', 'sentencepiece', 'scipy'],
      'tokenizer_class': 'LlamaTokenizerFast',
      'default_id': 'NousResearch/llama-2-7b-hf',
      'serialisation': 'safetensors',
      'model_ids': [
          'meta-llama/Llama-2-70b-chat-hf',
          'meta-llama/Llama-2-13b-chat-hf',
          'meta-llama/Llama-2-7b-chat-hf',
          'meta-llama/Llama-2-70b-hf',
          'meta-llama/Llama-2-13b-hf',
          'meta-llama/Llama-2-7b-hf',
          'NousResearch/llama-2-70b-chat-hf',
          'NousResearch/llama-2-13b-chat-hf',
          'NousResearch/llama-2-7b-chat-hf',
          'NousResearch/llama-2-70b-hf',
          'NousResearch/llama-2-13b-hf',
          'NousResearch/llama-2-7b-hf',
      ],
      'fine_tune_strategies': ({
          'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'
      },)
  }

  class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 12

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5

  def sanitize_parameters(self,
                          prompt: str,
                          top_k: int | None = None,
                          top_p: float | None = None,
                          temperature: float | None = None,
                          max_new_tokens: int | None = None,
                          use_default_prompt_template: bool = False,
                          use_llama2_prompt: bool = True,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE('v2' if use_llama2_prompt else 'v1') if use_default_prompt_template else None, use_default_prompt_template, **attrs), {
        'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k
    }, {}

  def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str:
    return generation_result[0]
