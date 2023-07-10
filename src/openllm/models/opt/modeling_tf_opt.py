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

import bentoml
import openllm

from .configuration_opt import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter
from ...utils import generate_labels


if t.TYPE_CHECKING:
    import transformers
else:
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")


logger = logging.getLogger(__name__)


class TFOPT(openllm.LLM["transformers.TFOPTForCausalLM", "transformers.GPT2Tokenizer"]):
    __openllm_internal__ = True

    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]] | None:
        tokenizer_kwds = {
            "padding_side": "left",
            "truncation_side": "left",
        }
        return {}, tokenizer_kwds

    def import_model(self, *args: t.Any, trust_remote_code: bool = False, **attrs: t.Any) -> bentoml.Model:
        _, tokenizer_attrs = self.llm_parameters

        config = transformers.AutoConfig.from_pretrained(self.model_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **tokenizer_attrs)
        tokenizer.pad_token_id = config.pad_token_id
        model: transformers.TFOPTForCausalLM = transformers.TFOPTForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=trust_remote_code, **attrs
        )
        return bentoml.transformers.save_model(
            self.tag,
            model,
            custom_objects={"tokenizer": tokenizer},
            labels=generate_labels(self),
        )

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        num_return_sequences: int | None = None,
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

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
        }
        return prompt_text, generation_config, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **attrs: t.Any) -> str:
        if len(generation_result) == 1:
            if self.config.format_outputs:
                logger.warning("'format_outputs' doesn't have any effect when 'num_return_sequences=1'")
            return generation_result[0]

        if self.config.format_outputs:
            return "Generated result:\n" + "\n -".join(generation_result)
        else:
            return "\n".join(generation_result)

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        input_ids = self.tokenizer(prompt, return_tensors="tf")
        generated_tensors = self.model.generate(
            **input_ids,
            do_sample=True,
            generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
        )
        return self.tokenizer.batch_decode(generated_tensors, skip_special_tokens=True)
