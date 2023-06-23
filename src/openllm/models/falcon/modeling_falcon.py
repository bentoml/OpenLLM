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

import importlib
import typing as t

import bentoml
import openllm
import transformers

from ..._prompt import default_formatter
from .configuration_falcon import DEFAULT_PROMPT_TEMPLATE


if t.TYPE_CHECKING:
    import torch
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")


class Falcon(openllm.LLM["transformers.TextGenerationPipeline", "transformers.PreTrainedTokenizerFast"]):
    __openllm_internal__ = True

    @property
    def import_kwargs(self):
        model_kwds = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
        }
        tokenizer_kwds: dict[str, t.Any] = {}
        return model_kwds, tokenizer_kwds

    def import_model(
        self, model_id: str, tag: bentoml.Tag, *model_args: t.Any, tokenizer_kwds: dict[str, t.Any], **attrs: t.Any
    ) -> bentoml.Model:
        trust_remote_code = attrs.pop("trust_remote_code", True)
        torch_dtype = attrs.pop("torch_dtype", torch.bfloat16)
        device_map = attrs.pop("device_map", "auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        try:
            return bentoml.transformers.save_model(
                tag,
                model,
                custom_objects={"tokenizer": tokenizer},
                external_modules=[importlib.import_module(model.__module__)],
            )
        finally:
            torch.cuda.empty_cache()

    def load_model(self, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any) -> t.Any:
        torch_dtype = attrs.pop("torch_dtype", torch.bfloat16)
        device_map = attrs.pop("device_map", "auto")
        trust_remote_code = attrs.pop("trust_remote_code", True)

        _ref = bentoml.transformers.get(tag)
        return transformers.pipeline(
            "text-generation",
            model=_ref.path,
            tokenizer=_ref.custom_objects["tokenizer"],
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **attrs,
        )

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        num_return_sequences: int | None = None,
        eos_token_id: int | None = None,
        use_default_prompt_template: bool = False,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            template_variables = default_formatter.extract_template_variables(DEFAULT_PROMPT_TEMPLATE)
            prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
            if "instruction" in prompt_variables:
                raise RuntimeError(
                    "'instruction' should be passed as the first argument instead of "
                    "kwargs when 'use_default_prompt_template=True'"
                )
            try:
                prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt, **prompt_variables)
            except KeyError as e:
                raise RuntimeError(
                    f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. "
                    "Use 'use_default_prompt_template=False' to disable the default prompt template."
                )
        else:
            prompt_text = prompt

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "eos_token_id": eos_token_id,
            **attrs,
        }

        return prompt_text, generation_config, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[dict[str, t.Any]], **_: t.Any) -> str:
        return "\n".join([i["generated_text"] for i in generation_result])

    def generate(self, prompt: str, **attrs: t.Any) -> list[dict[str, t.Any]]:
        eos_token_id = attrs.pop("eos_token_id", self.tokenizer.eos_token_id)

        # NOTE: our model here is the pipeline
        return self.model(
            prompt,
            do_sample=True,
            generation_config=self.config.model_construct_env(
                eos_token_id=eos_token_id, **attrs
            ).to_generation_config(),
        )
