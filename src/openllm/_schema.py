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
"""Schema definition for OpenLLM. This can be use for client interaction."""
from __future__ import annotations
import functools
import typing as t

import attr
import inflection

import openllm
from openllm._configuration import GenerationConfig
from openllm.utils import bentoml_cattr


if t.TYPE_CHECKING:
    from ._types import DictStrAny
else:
    DictStrAny = dict


@attr.frozen(slots=True)
class GenerationInput:
    prompt: str
    """The prompt to be sent to system."""

    llm_config: openllm.LLMConfig
    """A mapping of given LLM configuration values for given system."""

    @staticmethod
    def convert_llm_config(
        data: dict[str, t.Any] | openllm.LLMConfig, cls: type[openllm.LLMConfig] | None = None
    ) -> openllm.LLMConfig:
        if isinstance(data, openllm.LLMConfig):
            return data
        elif openllm.utils.LazyType(DictStrAny).isinstance(data):
            if cls is None:
                raise ValueError("'cls' must pass if given data is a dictionary.")
            return cls(**data)
        else:
            raise RuntimeError(f"Type {type(data)} is not yet supported.")

    @classmethod
    def for_model(cls, model_name: str, **attrs: t.Any) -> type[GenerationInput]:
        llm_config = openllm.AutoConfig.for_model(model_name, **attrs)
        return attr.make_class(
            inflection.camelize(llm_config["model_name"]) + "GenerationInput",
            attrs={
                "prompt": attr.field(type=str),
                "llm_config": attr.field(
                    type=llm_config.__class__,
                    default=llm_config,
                    converter=functools.partial(cls.convert_llm_config, cls=llm_config.__class__),
                ),
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

    @property
    def marshaled_config(self) -> GenerationConfig:
        return bentoml_cattr.structure(self.configuration, GenerationConfig)

    @property
    def unmarshaled(self) -> dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)


@attr.frozen(slots=True)
class MetadataOutput:
    model_id: str
    timeout: int
    model_name: str
    framework: str
    configuration: str


@attr.frozen(slots=True)
class EmbeddingsOutput:
    embeddings: t.List[float]
    num_tokens: int
