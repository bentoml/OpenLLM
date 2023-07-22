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

from .configuration_llama import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter


if t.TYPE_CHECKING:
    import torch
    import vllm

    import transformers
else:
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    vllm = openllm.utils.LazyLoader("vllm", globals(), "vllm")


logger = logging.getLogger(__name__)


class VLLMLlaMA(openllm.LLM["vllm.LLM", "transformers.LlamaTokenizerFast"]):
    __openllm_internal__ = True

    def sanitize_parameters(
        self,
        prompt: str,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        use_default_prompt_template: bool = True,
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

        generation_config = {"max_new_tokens": max_new_tokens, "temperature": temperature}

        return prompt_text, generation_config, {}

    @property
    def import_kwargs(self):
        model_kwds = {"device_map": "auto" if torch.cuda.device_count() > 1 else None}
        tokenizer_kwds: dict[str, t.Any] = {}
        return model_kwds, tokenizer_kwds

    def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str:
        return generation_result[0]

    def load_model(self, *args: t.Any, **attrs: t.Any) -> t.Any:
        model = transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path, *args, **attrs)
        if self.config.use_half_precision:
            model.half()
        return model

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        from ..._generation import StopOnTokens

        generation_kwargs = {
            "do_sample": True,
            "generation_config": self.config.model_construct_env(**attrs).to_generation_config(),
            "pad_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": transformers.StoppingCriteriaList([StopOnTokens()]),
        }

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            gen_tokens = self.model.generate(inputs.input_ids, **generation_kwargs)
            return self.tokenizer.batch_decode(gen_tokens)
