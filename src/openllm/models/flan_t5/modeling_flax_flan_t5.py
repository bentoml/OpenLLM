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


class FlaxFlanT5(openllm.LLM):
    __openllm_internal__ = True

    default_model: str = "google/flan-t5-large"

    variants = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]

    def preprocess_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs: t.Any,
    ) -> tuple[str, dict[str, t.Any]]:
        return prompt, self.config.with_options(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        ).model_dump(flatten=True)

    def postprocess_parameters(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
        return generation_result[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs: t.Any,
    ) -> list[str]:
        input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
        result_tensor = self.model.generate(
            input_ids,
            do_sample=True,
            generation_config=self.config.with_options(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs,
            ).to_generation_config(),
        )
        return self.tokenizer.batch_decode(
            result_tensor.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
