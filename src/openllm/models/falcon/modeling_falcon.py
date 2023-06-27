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
import platform
import typing as t

import bentoml
import openllm
import transformers
from bentoml._internal.frameworks.transformers import make_default_signatures
from bentoml._internal.models.model import ModelContext
from bentoml._internal.models.model import ModelOptions

from ..._prompt import default_formatter
from ...utils import is_flax_available
from ...utils import is_tf_available
from ...utils import is_torch_available
from ...utils import pkg
from .configuration_falcon import DEFAULT_PROMPT_TEMPLATE


if t.TYPE_CHECKING:
    import torch
    import torch.amp
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    torch.amp = openllm.utils.LazyLoader("torch.amp", globals(), "torch.amp")


class Falcon(openllm.LLM["transformers.PreTrainedModel", "transformers.PreTrainedTokenizerFast"]):
    __openllm_internal__ = True

    @property
    def import_kwargs(self):
        model_kwds = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        tokenizer_kwds: dict[str, t.Any] = {}
        return model_kwds, tokenizer_kwds

    def llm_post_init(self):
        self.device = torch.device("cuda")

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

        framework_versions = {"transformers": pkg.get_pkg_version("transformers")}
        if is_torch_available():
            framework_versions["torch"] = pkg.get_pkg_version("torch")
        if is_tf_available():
            from bentoml._internal.frameworks.utils.tensorflow import get_tf_version

            framework_versions[
                "tensorflow-macos" if platform.system() == "Darwin" else "tensorflow"
            ] = get_tf_version()
        if is_flax_available():
            framework_versions.update(
                {
                    "flax": pkg.get_pkg_version("flax"),
                    "jax": pkg.get_pkg_version("jax"),
                    "jaxlib": pkg.get_pkg_version("jaxlib"),
                }
            )

        try:
            with bentoml.models.create(
                tag,
                module="bentoml.transformers",
                api_version="v2",
                context=ModelContext(framework_name="transformers", framework_versions=framework_versions),
                options=ModelOptions(),
                signatures=make_default_signatures(model),
                external_modules=[
                    importlib.import_module(model.__module__),
                    importlib.import_module(tokenizer.__module__),
                ],
                metadata={"_pretrained_class": model.__class__.__name__, "_framework": model.framework},
            ) as bento_model:
                model.save_pretrained(bento_model.path)
                tokenizer.save_pretrained(bento_model.path)

                return bento_model
        finally:
            torch.cuda.empty_cache()

    def load_model(self, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any) -> t.Any:
        return transformers.AutoModelForCausalLM.from_pretrained(bentoml.models.get(tag).path, **attrs)

    def load_tokenizer(self, tag: bentoml.Tag, **attrs: t.Any) -> t.Any:
        trust_remote_code = attrs.pop("trust_remote_code", True)
        return transformers.AutoTokenizer.from_pretrained(
            bentoml.models.get(tag).path, trust_remote_code=trust_remote_code, **attrs
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

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
        return generation_result[0]

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        eos_token_id = attrs.pop("eos_token_id", self.tokenizer.eos_token_id)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.config.model_construct_env(
                    eos_token_id=eos_token_id, **attrs
                ).to_generation_config(),
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
