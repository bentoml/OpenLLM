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

import attr
import inflection

import openllm


@attr.frozen(slots=True)
class GenerationInput:
    prompt: str
    """The prompt to be sent to system."""

    llm_config: openllm.LLMConfig
    """A mapping of given LLM configuration values for given system."""

    @classmethod
    def for_model(cls, model_name: str, **attrs: t.Any) -> type[GenerationInput]:
        llm_config = openllm.AutoConfig.for_model(model_name, **attrs)
        return attr.make_class(
            inflection.camelize(llm_config.__openllm_model_name__) + "GenerationInput",
            attrs={
                "prompt": attr.field(type=str),
                "llm_config": attr.field(type=llm_config.__class__, default=llm_config),
            },
        )

    def model_dump(self) -> dict[str, t.Any]:
        return {"prompt": self.prompt, "llm_config": self.llm_config.model_dump(flatten=True)}


@attr.frozen(slots=True)
class GenerationOutput:
    responses: t.List[t.Any]
    """A list of responses from the system."""

    configuration: t.Dict[str, t.Any]
    """A mapping of configuration values for given system."""
