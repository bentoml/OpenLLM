from __future__ import annotations

import typing as t

import openllm

from ..._prompt import default_formatter
from .configuration_bart import DEFAULT_PROMPT_TEMPLATE


if t.TYPE_CHECKING:
    import transformers  # noqa


class TFFlanT5(openllm.LLM["transformers.AutoModelForSeq2SeqLM", "transformers.BartTokenizerFast"]):
    __openllm_internal__ = True

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool | None = None,
        early_stopping: bool | None = None,
        use_default_prompt_template: bool = True,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
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
            "use_cache": use_cache,
            "early_stopping": early_stopping,
        }
        return prompt_text, generation_config, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
        return generation_result[0]

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        input_ids = self.tokenizer(prompt, return_tensors="tf").input_ids
        outputs = self.model.generate(
            input_ids,
            generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
