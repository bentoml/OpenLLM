from __future__ import annotations
import logging
import typing as t
import openllm
from .configuration_llama import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import process_prompt

if t.TYPE_CHECKING: import ctransformers
logger = logging.getLogger(__name__)

class GGMLLlama(openllm.LLM["ctransformers.AutoModelForCausalLM", "ctransformers.AutoTokenizer"]):
    __openllm_internal__ = True

    def sanitize_parameters(self, prompt: str, top_k: int | None = None, top_p: float | None = None, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = True, use_llama2_prompt: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        _template = DEFAULT_PROMPT_TEMPLATE("v2" if use_llama2_prompt else "v1") if use_default_prompt_template else None
        return process_prompt(prompt, _template, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}, {}

    # FIXME: Have to see, what exactly to return
    def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str: return generation_result[0]

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        """This is how CTransformers work in most of the cases

        >> from ctransformers import AutoModelForCausalLM
        >> llm = AutoModelForCausalLM.from_pretrained('/path/to/ggml-gpt-2.bin', model_type='gpt2')

        so here we need to map llm to self.model as we are using ctransformers in our backend which
        is internally using ggml
        """
        return self.model(prompt)
