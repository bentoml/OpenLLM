from __future__ import annotations
import typing as t

import openllm_core
from openllm_core.prompts import PromptTemplate

DEFAULT_SYSTEM_MESSAGE = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
SINST_KEY, EINST_KEY, SYS_KEY, EOS_TOKEN, BOS_TOKEN = '[INST]', '[/INST]', '<<SYS>>', '</s>', '<s>'
# TODO: support history and v1 prompt implementation
_v1_prompt, _v2_prompt = (
  """{instruction}""",
  """{start_key} {sys_key}\n{system_message}\n{sys_key}\n\n{instruction}\n{end_key}\n""".format(
    start_key=SINST_KEY,
    sys_key=SYS_KEY,
    system_message='{system_message}',
    instruction='{instruction}',
    end_key=EINST_KEY,
  ),
)
PROMPT_MAPPING = {'v1': _v1_prompt, 'v2': _v2_prompt}


def _get_prompt(model_type: t.Literal['v1', 'v2']) -> PromptTemplate:
  return PromptTemplate(PROMPT_MAPPING[model_type])


DEFAULT_PROMPT_TEMPLATE = _get_prompt


class LlamaConfig(openllm_core.LLMConfig):
  """LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.

  It is a collection of foundation language models ranging from 7B to 65B parameters.

  Llama also include support for the recent propsed [Llama-2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

  Note that all variants of Llama including fine-tuning, quantisation format are all supported with ``openllm.Llama``.

  Refer to [Llama's model card](https://huggingface.co/docs/transformers/main/model_doc/llama)
  for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://github.com/facebookresearch/llama',
    'architecture': 'LlamaForCausalLM',
    'requirements': ['fairscale', 'sentencepiece', 'scipy'],
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
    'fine_tune_strategies': (
      {'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'},
    ),
  }

  class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 12

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5

  @property
  def default_prompt_template(self) -> str:
    return DEFAULT_PROMPT_TEMPLATE('v2').to_string()

  @property
  def default_system_message(self) -> str:
    return DEFAULT_SYSTEM_MESSAGE

  def sanitize_parameters(
    self,
    prompt: str,
    prompt_template: PromptTemplate | str | None = None,
    system_message: str | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    **attrs: t.Any,
  ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    system_message = DEFAULT_SYSTEM_MESSAGE if system_message is None else system_message
    if prompt_template is None:
      prompt_template = DEFAULT_PROMPT_TEMPLATE('v2')
    elif isinstance(prompt_template, str):
      prompt_template = PromptTemplate(template=prompt_template)
    return (
      prompt_template.with_options(system_message=system_message).format(instruction=prompt),
      {'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k},
      {},
    )
