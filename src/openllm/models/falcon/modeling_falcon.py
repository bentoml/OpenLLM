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

import bentoml
import transformers

import openllm

if t.TYPE_CHECKING:
    import torch
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")


class Falcon(openllm.LLM):
    __openllm_internal__ = True

    default_model = "tiiuae/falcon-7b"

    requirements = ["einops"]

    variants = ["tiiuae/falcon-7b", "tiiuae/falcon-40b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct"]

    import_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    def import_model(
        self, pretrained: str, tag: bentoml.Tag, *model_args: t.Any, tokenizer_kwds: dict[str, t.Any], **attrs: t.Any
    ) -> bentoml.Model:
        trust_remote_code = attrs.pop("trust_remote_code", True)
        torch_dtype = attrs.pop("torch_dtype", torch.bfloat16)
        device_map = attrs.pop("device_map", "auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype, device_map=device_map
        )
        config = transformers.AutoConfig.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        transformers.AutoModelForCausalLM.register(config.__class__, model.__class__)
        return bentoml.transformers.save_model(
            str(tag),
            transformers.pipeline("text-generation", model=model, tokenizer=tokenizer),
            custom_objects={"tokenizer": tokenizer},
        )

    def preprocess_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        num_return_sequences: int | None = None,
        eos_token_id: int | None = None,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any]]:
        generation_config = self.config.model_construct_env(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_token_id,
            **attrs,
        ).model_dump(flatten=True)

        return prompt, generation_config

    def postprocess_parameters(self, prompt: str, generation_result: t.Sequence[dict[str, t.Any]], **_: t.Any) -> str:
        return "\n".join([i["generated_text"] for i in generation_result])

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        num_return_sequences: int | None = None,
        eos_token_id: int | None = None,
        **attrs: t.Any,
    ) -> list[dict[str, t.Any]]:
        # NOTE: MK tokenizer into this pipeline, since we can't inject tokenizer at load time
        self.model.tokenizer = self.tokenizer
        # NOTE: our model here is the pipeline
        return self.model(
            prompt,
            do_sample=True,
            generation_config=self.config.model_construct_env(
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                eos_token_id=eos_token_id or self.tokenizer.eos_token_id,
                **attrs,
            ).to_generation_config(),
        )
