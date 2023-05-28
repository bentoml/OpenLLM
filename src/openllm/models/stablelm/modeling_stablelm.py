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

import logging
import typing as t

import openllm

from .configuration_stablelm import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING:
    import torch
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")


logger = logging.getLogger(__name__)


class StableLM(openllm.LLM):
    __openllm_internal__ = True

    default_model = "StabilityAI/stablelm-tuned-alpha-3b"

    variants = [
        "StabilityAI/stablelm-tuned-alpha-3b",
        "StabilityAI/stablelm-tuned-alpha-7b",
        "StabilityAI/stablelm-base-alpha-3b",
        "StabilityAI/stablelm-base-alpha-7b",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_parameters(
        self,
        prompt: str,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any]]:
        if "tuned" in self.default_model:
            prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt)
        else:
            prompt_text = prompt

        return prompt_text, self.config.model_construct_env(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **attrs,
        ).model_dump(flatten=True)

    def postprocess_parameters(self, prompt: str, generation_result: str, **_: t.Any) -> str:
        return generation_result

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        **attrs: t.Any,
    ) -> list[str]:
        if torch.cuda.is_available():
            self.model.half().cuda()

        inputs = t.cast("torch.Tensor", self.tokenizer(prompt, return_tensors="pt")).to(self.device)
        with torch.device(self.device):
            result_tensor = self.model.generate(
                **inputs,
                do_sample=True,
                generation_config=self.config.model_construct_env(
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    **attrs,
                ).to_generation_config(),
            )
        return self.tokenizer.decode(result_tensor[0], skip_special_tokens=True)
