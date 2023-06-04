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

from transformers import StoppingCriteria, StoppingCriteriaList

import openllm

from ..._prompt import default_formatter
from .configuration_stablelm import DEFAULT_PROMPT_TEMPLATE


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = set([50278, 50279, 50277, 1, 0])
        return input_ids[0][-1] in stop_ids


if t.TYPE_CHECKING:
    import torch
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")


logger = logging.getLogger(__name__)


class StableLM(openllm.LLM):
    __openllm_internal__ = True

    load_in_mha = False

    default_model = "StabilityAI/stablelm-tuned-alpha-3b"

    variants = [
        "StabilityAI/stablelm-tuned-alpha-3b",
        "StabilityAI/stablelm-tuned-alpha-7b",
        "StabilityAI/stablelm-base-alpha-3b",
        "StabilityAI/stablelm-base-alpha-7b",
    ]

    import_kwargs = {
        "torch_dtype": torch.float16,
        "load_in_8bit": True if torch.cuda.is_available() and torch.cuda.device_count() == 1 else False,
        "device_map": "auto",
    }

    def sanitize_parameters(
        self,
        prompt: str,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        use_default_prompt_template: bool = True,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if "tuned" in self._pretrained and use_default_prompt_template:
            prompt_variables = {
                k: v
                for k, v in attrs.items()
                if k in default_formatter.extract_template_variables(DEFAULT_PROMPT_TEMPLATE)
            }
            if "instruction" in prompt_variables:
                raise RuntimeError(
                    "'instruction' should be passed as the first argument "
                    "instead of kwargs when 'use_default_prompt_template=True'"
                )
            prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt, **prompt_variables)
        else:
            prompt_text = prompt

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }

        return prompt_text, generation_config, {}

    def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str:
        return generation_result[0]

    @torch.inference_mode()
    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        if torch.cuda.is_available():
            self.model.cuda()
        if not self.model.is_loaded_in_8bit:
            self.model.half()

        generation_kwargs = {
            "do_sample": True,
            "generation_config": self.config.model_construct_env(**attrs).to_generation_config(),
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),
        }

        inputs = t.cast("torch.Tensor", self.tokenizer(prompt, return_tensors="pt")).to(self.model.device)
        tokens = self.model.generate(**inputs, **generation_kwargs)
        return [self.tokenizer.decode(tokens[0], skip_special_tokens=True)]
