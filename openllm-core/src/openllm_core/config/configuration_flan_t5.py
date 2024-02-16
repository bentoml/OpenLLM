from __future__ import annotations

import openllm_core, pydantic
from openllm_core._configuration import ModelSettings


class FlanT5Config(openllm_core.LLMConfig):
  """FLAN-T5 was released in the paper [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf).

  It is an enhanced version of T5 that has been finetuned in a mixture of tasks.

  Refer to [FLAN-T5's page](https://huggingface.co/docs/transformers/model_doc/flan-t5) for more information.
  """

  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  metadata_config: ModelSettings = pydantic.Field(
    default={
      'url': 'https://huggingface.co/docs/transformers/model_doc/flan-t5',
      'architecture': 'T5ForConditionalGeneration',
      'model_type': 'seq2seq_lm',
      'backend': ('pt',),
      # NOTE: See https://www.philschmid.de/fine-tune-flan-t5. No specific template found, but seems to have the same dialogue style
      'default_id': 'google/flan-t5-large',
      'model_ids': [
        'google/flan-t5-small',
        'google/flan-t5-base',
        'google/flan-t5-large',
        'google/flan-t5-xl',
        'google/flan-t5-xxl',
      ],
    },
    repr=False,
    exclude=True,
  )

  generation_config: openllm_core.GenerationConfig = pydantic.Field(
    default=openllm_core.GenerationConfig(
      temperature=0.9, max_new_tokens=2048, top_k=50, top_p=0.4, repetition_penalty=1.0
    )
  )

  @property
  def template(self) -> str:
    return 'Answer the following question:\nQuestion: {instruction}\nAnswer:'
