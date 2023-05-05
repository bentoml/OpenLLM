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
from .configuration_flan_t5 import FlanT5Config as FlanT5Config

if t.TYPE_CHECKING:
    import bentoml
    import jax.numpy as jnp
    import transformers

else:
    jnp = openllm.utils.LazyLoader("jnp", globals(), "jax.numpy")
    bentoml = openllm.utils.LazyLoader("bentoml", globals(), "bentoml")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")


def import_model(
    pretrained_or_path: str,
    model_kwargs: dict[str, t.Any] | None = None,
    tokenizer_kwargs: dict[str, t.Any] | None = None,
    config_kwargs: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """Import any Flax Flan-T5 pretrained models weight into the store.

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

    tag = openllm.utils.generate_tag_from_model_name(pretrained_or_path, prefix="flax", **config_kwargs)
    try:
        return bentoml.transformers.get(tag)
    except bentoml.exceptions.NotFound:
        model = transformers.FlaxT5ForConditionalGeneration.from_pretrained(pretrained_or_path, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_or_path, **tokenizer_kwargs)
        return bentoml.transformers.save_model(str(tag), model, custom_objects={"tokenizer": tokenizer})


class FlaxFlanT5(
    LLMRunnable[transformers.FlaxT5ForConditionalGeneration, transformers.T5TokenizerFast], start_model_name="flan-t5"
):
    default_model: str = "google/flan-t5-large"
    config_class = FlanT5Config

    ATTACH_TOKENIZER = False

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
        input_ids: jnp.ndarray,
        max_length: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs: t.Any,
    ) -> jnp.ndarray:
        return self.model.generate(
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


class FlaxFlanT5WithTokenizer(FlaxFlanT5, start_model_name="flan-t5"):
    default_model: str = "google/flan-t5-large"

    ATTACH_TOKENIZER = True

    def _generate(self, prompt: str, **kwargs: t.Any) -> list[str]:
        input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
        outputs = super()._generate(input_ids, **kwargs)
        return self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
