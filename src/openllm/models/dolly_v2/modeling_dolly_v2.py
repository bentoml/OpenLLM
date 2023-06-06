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
import transformers

import openllm

from .configuration_dolly_v2 import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING:
    import torch

    from ..._types import LLMTokenizer
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)


def get_special_token_id(tokenizer: LLMTokenizer, key: str) -> int:
    """
    Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer: the tokenizer
        key: the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


class DollyV2(openllm.LLM):
    __openllm_internal__ = True

    default_model = "databricks/dolly-v2-3b"

    load_in_mha = False  # NOTE: disable bettertransformer for dolly

    pretrained = ["databricks/dolly-v2-3b", "databricks/dolly-v2-7b", "databricks/dolly-v2-12b"]

    import_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "_tokenizer_padding_side": "left"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def import_model(
        self, pretrained: str, tag: bentoml.Tag, *model_args: t.Any, tokenizer_kwds: dict[str, t.Any], **attrs: t.Any
    ) -> bentoml.Model:
        trust_remote_code = attrs.pop("trust_remote_code", True)
        torch_dtype = attrs.pop("torch_dtype", torch.bfloat16)
        device_map = attrs.pop("device_map", "auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained, **tokenizer_kwds)
        pipeline = transformers.pipeline(
            model=pretrained,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        return bentoml.transformers.save_model(
            tag,
            pipeline,
            custom_objects={"tokenizer": tokenizer},
            external_modules=[importlib.import_module(pipeline.__module__)],
        )

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        use_default_prompt_template: bool = False,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt)
        else:
            prompt_text = prompt

        # NOTE: The rest of attrs should be kwargs for GenerationConfig
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "use_default_prompt_template": use_default_prompt_template,
            **attrs,
        }

        return prompt_text, generate_kwargs, {}

    def postprocess_generate(self, prompt: str, generation_result: str, **_: t.Any) -> str:
        return generation_result

    @torch.inference_mode()
    def generate(self, prompt: str, **attrs: t.Any) -> str:
        self.model.tokenizer = self.tokenizer
        llm_config: openllm.DollyV2Config = self.config.model_construct_env(**attrs)
        decoded = self.model(prompt, do_sample=True, generation_config=llm_config.to_generation_config())

        # If the full text is requested, then append the decoded text to the original instruction.
        # This technically isn't the full text, as we format the instruction in the prompt the model has been
        # trained on, but to the client it will appear to be the full text.
        if llm_config.return_full_text:
            decoded = f"{DEFAULT_PROMPT_TEMPLATE.format(prompt)}\n{decoded}"

        return decoded
