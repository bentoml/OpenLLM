# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import typing as t

import openllm

from ...runner_utils import LLMRunnable
from .configuration_flan_t5 import FlanT5Config

if t.TYPE_CHECKING:
    import bentoml
    import tensorflow as tf
    import transformers
else:
    tf = openllm.utils.LazyLoader("tf", globals(), "tensorflow")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")
    bentoml = openllm.utils.LazyLoader("bentoml", globals(), "bentoml")


def import_model(
    pretrained_or_path: str,
    model_kwargs: dict[str, t.Any] | None = None,
    tokenizer_kwargs: dict[str, t.Any] | None = None,
    config_kwargs: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """Import any PyTorch Flan-T5 pretrained models weight into the store.

    Args:
        model_name: The name of the model to import.
        model_kwargs: Additional kwargs to pass to the ``transformers.AutoModelForSeq2SeqLM`` constructors.
        tokenizer_kwargs: Additional kwargs to pass to the ``transformers.AutoTokenizer`` constructors.
        config_kwargs: Additional kwargs to pass to the ``transformers.AutoConfig`` constructors to determine the model tag.

    Returns:
        a ``bentoml.Model`` instance.
    """
    model_kwargs = model_kwargs or {}
    tokenizer_kwargs = tokenizer_kwargs or {}
    config_kwargs = config_kwargs or {}

    tag = openllm.utils.generate_tag_from_model_name(pretrained_or_path, prefix="tf", **config_kwargs)
    try:
        return bentoml.transformers.get(tag)
    except bentoml.exceptions.NotFound:
        model = transformers.TFT5ForConditionalGeneration.from_pretrained(pretrained_or_path, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_or_path, **tokenizer_kwargs)
        return bentoml.transformers.save_model(str(tag), model, custom_objects={"tokenizer": tokenizer})


class TFFlanT5(LLMRunnable, start_model_name="flan-t5"):
    default_model: str = "google/flan-t5-large"
    config_class = FlanT5Config

    ATTACH_TOKENIZER = True

    _llm_config: FlanT5Config

    variants = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]

    def _generate(
        self,
        prompt: str,
        max_length: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs: t.Any,
    ) -> tf.Tensor:
        input_ids: tf.Tensor = self.tokenizer(prompt, return_tensors="tf").input_ids
        outputs = self.model.generate(
            input_ids,
            max_length=max_length if max_length is not None else self._llm_config.max_length,
            do_sample=do_sample,
            temperature=temperature if temperature is not None else self._llm_config.temperature,
            top_k=top_k if top_k is not None else self._llm_config.top_k,
            top_p=top_p if top_p is not None else self._llm_config.top_p,
            repetition_penalty=repetition_penalty
            if repetition_penalty is not None
            else self._llm_config.repetition_penalty,
            **kwargs,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
