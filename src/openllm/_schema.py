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
"""
Schema definition for OpenLLM. This can be use for client interaction.
"""
from __future__ import annotations

import string
import typing as t

import inflection
import pydantic

import openllm


class PromptFormatter(string.Formatter):
    def vformat(self, format_string: str, args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]) -> str:
        if len(args) > 0:
            raise ValueError("Positional arguments are not supported")
        return super().vformat(format_string, args, kwargs)

    def check_unused_args(
        self, used_args: set[int | str], args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]
    ) -> None:
        """Check if extra params is passed."""
        extras = set(kwargs).difference(used_args)
        if extras:
            raise KeyError(f"Extra params passed: {extras}")

    def extract_template_variables(self, template: str) -> t.Sequence[str]:
        """Extract template variables from a template string."""
        return [field[1] for field in self.parse(template) if field[1] is not None]


# TODO: Support jinja2 template, go template and possible other prompt template engine.
_default_formatter = PromptFormatter()


class PromptTemplate(pydantic.BaseModel):
    template: str
    input_variables: t.Sequence[str]

    model_config = {"extra": "forbid"}

    def to_str(self, **kwargs: str) -> str:
        """Generate a prompt from the template and input variables"""
        if not kwargs:
            raise ValueError("Keyword arguments are required")
        if not all(k in kwargs for k in self.input_variables):
            raise ValueError(f"Missing required input variables: {self.input_variables}")
        return _default_formatter.format(self.template, **kwargs)

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        input_variables = _default_formatter.extract_template_variables(template)
        return cls(template=template, input_variables=input_variables)

    @classmethod
    def from_default(cls, model: str) -> PromptTemplate:
        template = getattr(openllm.utils.get_lazy_module(model), "DEFAULT_PROMPT_TEMPLATE")
        if template is None:
            raise ValueError(f"Model {model} does not have a default prompt template.")
        return cls.from_template(template)


class GenerationInput(pydantic.BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str
    """The prompt to be sent to system."""

    llm_config: openllm.LLMConfig
    """A mapping of given LLM configuration values for given system."""

    @classmethod
    def for_model(cls, model_name: str, **kwds: t.Any) -> type[GenerationInput]:
        llm_config = openllm.AutoConfig.for_model(model_name, **kwds)
        return pydantic.create_model(
            inflection.camelize(llm_config.__openllm_model_name__) + "GenerationInput",
            __base__=(cls,),
            __module__=llm_config.__module__,
            prompt=(str, ...),
            llm_config=(llm_config.__class__, ...),
        )


class GenerationOutput(pydantic.BaseModel):
    model_config = {"extra": "forbid"}

    responses: t.List[str]
    """A list of responses from the system."""

    configuration: t.Dict[str, t.Any]
    """A mapping of configuration values for given system."""
