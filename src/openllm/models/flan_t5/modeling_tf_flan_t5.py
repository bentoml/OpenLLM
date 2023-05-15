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


class TFFlanT5(openllm.LLM, implementation="tf", _internal=True):
    default_model: str = "google/flan-t5-large"

    variants = [
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]

    def generate(
        self,
        prompt: str,
        max_length: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        **kwargs: t.Any,
    ) -> list[str]:
        generation_kwargs = self.config.with_options(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        ).dict()["generation_config"]
        input_ids = self.tokenizer(prompt, return_tensors="tf").input_ids
        outputs = self.model.generate(input_ids, do_sample=do_sample, **generation_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
