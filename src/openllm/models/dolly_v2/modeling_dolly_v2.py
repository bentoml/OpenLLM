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
import logging
import typing as t

import bentoml
import openllm
import transformers

from .configuration_dolly_v2 import DEFAULT_PROMPT_TEMPLATE


if t.TYPE_CHECKING:
    import torch
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)


class DollyV2(openllm.LLM["transformers.Pipeline", "transformers.PreTrainedTokenizer"]):
    __openllm_internal__ = True

    @property
    def import_kwargs(self):
        model_kwds = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        tokenizer_kwds = {"padding_side": "left"}
        return model_kwds, tokenizer_kwds

    def llm_post_init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def import_model(
        self, model_id: str, tag: bentoml.Tag, *model_args: t.Any, tokenizer_kwds: dict[str, t.Any], **attrs: t.Any
    ) -> bentoml.Model:
        trust_remote_code = attrs.pop("trust_remote_code", True)
        torch_dtype = attrs.pop("torch_dtype", torch.bfloat16)
        device_map = attrs.pop("device_map", "auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **tokenizer_kwds)
        pipeline = transformers.pipeline(
            model=model_id,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        try:
            return bentoml.transformers.save_model(
                tag,
                pipeline,
                custom_objects={"tokenizer": tokenizer},
                external_modules=[importlib.import_module(pipeline.__module__)],
            )
        finally:
            if openllm.utils.is_torch_available() and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        # NOTE: The rest of attrs should be kwargs for GenerationConfig
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            **attrs,
        }

        return prompt, generate_kwargs, {}

    def postprocess_generate(
        self, prompt: str, generation_result: list[dict[t.Literal["generated_text"], str]], **_: t.Any
    ) -> str:
        return generation_result[0]["generated_text"]

    def generate(self, prompt: str, **attrs: t.Any) -> list[dict[t.Literal["generated_text"], str]]:
        with torch.inference_mode():
            self.model.tokenizer = self.tokenizer
            llm_config = self.config.model_construct_env(**attrs)
            decoded: list[dict[t.Literal["generated_text"], str]] = self.model(
                prompt, generation_config=llm_config.to_generation_config()
            )

            if llm_config.return_full_text:
                return [
                    {k: f"{DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt)}\n{generated}"}
                    for i in decoded
                    for k, generated in i.items()
                ]

            return decoded
