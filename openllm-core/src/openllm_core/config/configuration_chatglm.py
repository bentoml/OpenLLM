from __future__ import annotations
import typing as t

import openllm_core

from openllm_core.utils import dantic

START_CHATGLM_COMMAND_DOCSTRING = '''\
Run a LLMServer for ChatGLM model.

\b
> See more information about ChatGLM at [THUDM/ChatGLM-6b](https://huggingface.co/thudm/chatglm-6b)

\b
## Usage

Currently, ChatGLM only supports PyTorch. Make sure ``torch`` is available in your system.

\b
ChatGLM Runner will use THUDM/ChatGLM-6b as the default model. To change to any other ChatGLM
saved pretrained, or a fine-tune ChatGLM, provide ``OPENLLM_CHATGLM_MODEL_ID='thudm/chatglm-6b-int8'``
or provide `--model-id` flag when running ``openllm start chatglm``:

\b
$ openllm start chatglm --model-id='thudm/chatglm-6b-int8'
'''
DEFAULT_PROMPT_TEMPLATE = '''{instruction}'''

class ChatGLMConfig(openllm_core.LLMConfig):
  """ChatGLM is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM) framework.

  With the quantization technique, users can deploy locally on consumer-grade graphics cards
  (only 6GB of GPU memory is required at the INT4 quantization level).

  ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue.
  The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning,
  feedback bootstrap, and reinforcement learning wit human feedback.
  With only about 6.2 billion parameters, the model is able to generate answers that are in line
  with human preference.

  Refer to [ChatGLM's GitHub page](https://github.com/THUDM/ChatGLM-6B) for more information.
  """
  __config__ = {
      'name_type': 'lowercase',
      'trust_remote_code': True,
      'timeout': 3600000,
      'requires_gpu': True,
      'url': 'https://github.com/THUDM/ChatGLM-6B',
      'requirements': ['cpm-kernels', 'sentencepiece'],
      'architecture': 'ChatGLMForConditionalGeneration',
      'default_id': 'thudm/chatglm-6b',
      'model_ids': ['thudm/chatglm-6b', 'thudm/chatglm-6b-int8', 'thudm/chatglm-6b-int4', 'thudm/chatglm2-6b', 'thudm/chatglm2-6b-int4']
  }
  retain_history: bool = dantic.Field(False, description='Whether to retain history given to the model. If set to True, then the model will retain given history.')
  use_half_precision: bool = dantic.Field(True, description='Whether to use half precision for model.')

  class GenerationConfig:
    max_new_tokens: int = 2048
    num_beams: int = 1
    top_p: float = 0.7
    temperature: float = 0.95

  def sanitize_parameters(self,
                          prompt: str,
                          max_new_tokens: int | None = None,
                          num_beams: int | None = None,
                          top_p: float | None = None,
                          temperature: float | None = None,
                          chat_history: list[tuple[str, str]] | None = None,
                          use_default_prompt_template: bool = False,
                          **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    prompt_text = ''
    if use_default_prompt_template and chat_history is not None:
      for i, (old_query, response) in enumerate(chat_history):
        prompt_text += f'[Round {i}]\n问:{old_query}\n答:{response}\n'
      prompt_text += f'[Round {len(chat_history)}]\n问:{prompt}\n答:'
    else:
      prompt_text = prompt
    postprocess_generate_kwargs = {'chat_history': chat_history if chat_history is not None else None}
    return prompt_text, {'max_new_tokens': max_new_tokens, 'num_beams': num_beams, 'top_p': top_p, 'temperature': temperature, **attrs}, postprocess_generate_kwargs

  def postprocess_generate(self, prompt: str, generation_result: tuple[str, list[tuple[str, str]]], *, chat_history: list[tuple[str, str]] | None = None, **attrs: t.Any) -> str:
    generated, history = generation_result
    if self.config.retain_history:
      if chat_history is None: raise ValueError("'retain_history' is True while there is no history provided.")
      chat_history.extend(history)
    return generated
