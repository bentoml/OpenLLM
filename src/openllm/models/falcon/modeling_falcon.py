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

from .configuration_falcon import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter


if t.TYPE_CHECKING:
    import torch

    import transformers
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")


class Falcon(openllm.LLM["transformers.PreTrainedModel", "transformers.PreTrainedTokenizerBase"]):
    __openllm_internal__ = True

    @property
    def import_kwargs(self):
        model_kwds = {"torch_dtype": torch.bfloat16, "device_map": "auto" if torch.cuda.is_available() else None}
        tokenizer_kwds: dict[str, t.Any] = {}
        return model_kwds, tokenizer_kwds

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
                ) from None
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

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
        return generation_result[0]

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        eos_token_id = attrs.pop("eos_token_id", self.tokenizer.eos_token_id)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.config.model_construct_env(
                    eos_token_id=eos_token_id, **attrs
                ).to_generation_config(),
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_one(
        self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any
    ) -> list[dict[t.Literal["generated_text"], str]]:
        from ..._generation import StopSequenceCriteria

        max_new_tokens = preprocess_generate_kwds.pop("max_new_tokens", 200)
        encoded_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        src_len = encoded_inputs["input_ids"].shape[1]
        stopping_criteria = preprocess_generate_kwds.pop("stopping_criteria", transformers.StoppingCriteriaList([]))
        stopping_criteria.append(StopSequenceCriteria(stop, self.tokenizer))
        outputs = self.model.generate(
            encoded_inputs["input_ids"], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria
        )

        result = self.tokenizer.decode(outputs[0].tolist()[src_len:])
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return [{"generated_text": result}]
