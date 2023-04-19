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

from ...runner_utils import (LLMRunnable, assign_start_model_name,
                             generate_tokenizer_runner)
from .configuration_flan_t5 import FlanT5Config

if t.TYPE_CHECKING:
    import bentoml
    import torch
    import transformers
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
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

    tag = openllm.utils.generate_tag_from_model_name(pretrained_or_path, **config_kwargs)

    try:
        return bentoml.transformers.get(tag)
    except bentoml.exceptions.NotFound:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_or_path, **model_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_or_path, **tokenizer_kwargs)
        return bentoml.transformers.save_model(str(tag), model, custom_objects={"tokenizer": tokenizer})


def _FlanT5Tokenizer(
    pretrained_or_path: str | None = None, embedded: bool = True, **kwargs: t.Any
) -> openllm.types.TokenizerRunner:
    """Get the runner for the tokenizer.

    Args:
        model_name: The name of the FLAN-T5 model to import.
        embedded: Whether to use the embedded runner or not.
        **kwargs: Additional kwargs to pass to the ``transformers.AutoTokenizer`` constructors.

    Returns:
        The runner for the tokenizer.
    """
    if pretrained_or_path is None:
        pretrained_or_path = FlanT5.default_model

    return generate_tokenizer_runner(
        import_model(pretrained_or_path, **kwargs).custom_objects["tokenizer"], embedded=embedded
    )


FlanT5Tokenizer = assign_start_model_name("flan-t5")(_FlanT5Tokenizer)


class FlanT5(
    LLMRunnable[transformers.T5ForConditionalGeneration, transformers.T5TokenizerFast], start_model_name="flan-t5"
):
    default_model: str = "google/flan-t5-large"
    config_class = FlanT5Config

    ATTACH_TOKENIZER = False

    _llm_config: FlanT5Config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def pretrained_models() -> list[str]:
        return [
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
        ]

    def _generate(
        self,
        input_ids: torch.Tensor,
        max_length: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs: t.Any,
    ) -> torch.Tensor:
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


class FlanT5WithTokenizer(FlanT5, start_model_name="flan-t5"):
    default_model: str = "google/flan-t5-large"

    ATTACH_TOKENIZER = True

    def _generate(self, prompt: str, **kwargs: t.Any) -> list[str]:
        input_ids: torch.Tensor = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        outputs = super()._generate(input_ids, **kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
