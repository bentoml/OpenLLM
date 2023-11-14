from __future__ import annotations
import typing as t

import openllm_core
from openllm_core.prompts import PromptTemplate

START_MISTRAL_COMMAND_DOCSTRING = """\
Run a LLMServer for Mistral model.

\b
> See more information about Mistral at [Mistral's model card](https://huggingface.co/docs/transformers/main/model_doc/mistral

\b
## Usage

By default, this model will use [vLLM](https://github.com/vllm-project/vllm) for inference.
This model will also supports PyTorch.

\b
- To use PyTorch, set the environment variable ``OPENLLM_BACKEND="pt"``

\b
- To use vLLM, set the environment variable ``OPENLLM_BACKEND="vllm"``

\b
Mistral Runner will use mistralai/Mistral-7B-Instruct-v0.1, as the default model.
To change to any other Mistral saved pretrained, or a fine-tune Mistral,
provide ``OPENLLM_MODEL_ID='openlm-research/open_mistral_7b'`` or provide
`--model-id` flag when running ``openllm start mistral``:

\b
$ openllm start mistral --model-id 'mistralai/Mistral-7B-v0.1'

"""

# https://docs.mistral.ai/usage/guardrailing/
DEFAULT_SYSTEM_MESSAGE = """Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."""
SINST_KEY, EINST_KEY, BOS_TOKEN, EOS_TOKEN = '[INST]', '[/INST]', '<s>', '</s>'
DEFAULT_PROMPT_TEMPLATE = """{start_key}{start_inst} {system_message} {instruction} {end_inst}\n""".format(
  start_inst=SINST_KEY,
  end_inst=EINST_KEY,
  start_key=BOS_TOKEN,
  system_message='{system_message}',
  instruction='{instruction}',
)


class MistralConfig(openllm_core.LLMConfig):
  """Mistral-7B-v0.1 is a small, yet powerful model adaptable to many use-cases. It is released under Apache 2.0 licence, and it's easy to deploy on any cloud

  It currently contains a foundation model and a instruct-finetuned model.

  Refer to [Mistral's HuggingFace page](https://huggingface.co/docs/transformers/main/model_doc/mistral)
  for more information.
  """

  __config__ = {
    'name_type': 'lowercase',
    'url': 'https://mistral.ai',
    'architecture': 'MistralForCausalLM',
    'add_generation_prompt': True,
    'default_id': 'mistralai/Mistral-7B-Instruct-v0.1',
    'serialisation': 'safetensors',
    # NOTE: see https://docs.mistral.ai/usage/guardrailing/
    # and https://docs.mistral.ai/llm/mistral-instruct-v0.1
    'model_ids': ['mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-v0.1'],
    'fine_tune_strategies': (
      {'adapter_type': 'lora', 'r': 64, 'lora_alpha': 16, 'lora_dropout': 0.1, 'bias': 'none'},
    ),
  }

  class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40

  class SamplingParams:
    best_of: int = 1
    presence_penalty: float = 0.5

  @property
  def default_prompt_template(self) -> str:
    return DEFAULT_PROMPT_TEMPLATE

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
      prompt_template = DEFAULT_PROMPT_TEMPLATE
    elif isinstance(prompt_template, str):
      prompt_template = PromptTemplate(template=prompt_template)

    return (
      prompt_template.with_options(system_message=system_message).format(instruction=prompt),
      {'max_new_tokens': max_new_tokens, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k},
      {},
    )
