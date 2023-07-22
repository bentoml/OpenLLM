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

from .configuration_baichuan import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter


if t.TYPE_CHECKING:
    import torch

    import transformers
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")


class Baichuan(openllm.LLM["transformers.PreTrainedModel", "transformers.PreTrainedTokenizerBase"]):
    __openllm_internal__ = True

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        use_default_prompt_template: bool = False,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            template_variables = default_formatter.extract_template_variables(DEFAULT_PROMPT_TEMPLATE)
            prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
            if "instruction" in prompt_variables:
                raise RuntimeError(
                    "'instruction' should be passed as the first argument "
                    "instead of kwargs when 'use_default_prompt_template=True'"
                )
            try:
                prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt, **prompt_variables)
            except KeyError as e:
                raise RuntimeError(
                    f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. "
                    "Use 'use_default_prompt_template=False' to disable the default prompt template."
                ) from None
        else:
            prompt_text = prompt
        # NOTE: The rest of attrs should be kwargs for GenerationConfig
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperature": temperature,
            **attrs,
        }

        return prompt_text, generate_kwargs, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
        return generation_result[0]

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model.generate(
                **inputs,
                generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
