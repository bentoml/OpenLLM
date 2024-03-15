from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings

SINST_KEY, EINST_KEY, SYS_KEY, EOS_TOKEN, BOS_TOKEN = '[INST]', '[/INST]', '<<SYS>>', '</s>', '<s>'


class LlamaConfig(openllm_core.LLMConfig):
  """LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.

  It is a collection of foundation language models ranging from 7B to 65B parameters.

  Llama also include support for the recent propsed [Llama-2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

  Note that all variants of Llama including fine-tuning, quantisation format are all supported with ``openllm.Llama``.

  Refer to [Llama's model card](https://huggingface.co/docs/transformers/main/model_doc/llama)
  for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'name_type': 'lowercase',
      'url': 'https://github.com/facebookresearch/llama',
      'architecture': 'LlamaForCausalLM',
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
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig.model_construct(
      max_new_tokens=128, temperature=0.6, top_p=0.9, top_k=12, best_of=1, presence_penalty=0.5
    )
  )

  @property
  def template(self) -> str:
    return '{start_key} {sys_key}\n{system_message}\n{sys_key}\n\n{instruction}\n{end_key}\n'.format(
      start_key=SINST_KEY,
      sys_key=SYS_KEY,
      system_message='{system_message}',
      instruction='{instruction}',
      end_key=EINST_KEY,
    )

  @property
  def system_message(self) -> str:
    return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
