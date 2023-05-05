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

import typing as t
from abc import ABC

import pydantic

import openllm


class PromptTemplate(pydantic.BaseModel):
    template: str
    input_variables: t.Sequence[str]

    class Config:
        extra = "forbid"

    def to_str(self, **kwargs: str) -> str:
        """Generate a prompt from the template and input variables"""
        if not kwargs:
            raise ValueError("Keyword arguments are required")
        if not all(k in kwargs for k in self.input_variables):
            raise ValueError(f"Missing required input variables: {self.input_variables}")
        return openllm.prompts.default_formatter.format(self.template, **kwargs)

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        input_variables = openllm.prompts.default_formatter.extract_template_variables(template)
        return cls(template=template, input_variables=input_variables)

    @classmethod
    def from_default(cls, model: str) -> PromptTemplate:
        template = getattr(openllm.utils.get_lazy_module(model), "DEFAULT_PROMPT_TEMPLATE")
        if template is None:
            raise ValueError(f"Model {model} does not have a default prompt template.")
        return cls.from_template(template)


class BaseIO(pydantic.BaseModel, ABC):
    class Config:
        extra = "forbid"


class PromptInput(BaseIO):
    prompt: str
    """The prompt to be sent to system."""

    llm_config: t.Dict[str, t.Any]
    """A mapping of given LLM configuration values for given system."""


class PromptOutput(BaseIO):
    responses: t.List[str]
    """A list of responses from the system."""

    configuration: t.Dict[str, t.Any]
    """A mapping of configuration values for given system."""
