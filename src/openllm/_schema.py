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

from ._configuration import GenerationConfig
from ._configuration import LLMConfig
from .utils import LazyLoader
from .utils import LazyType
from .utils import bentoml_cattr
from .utils import requires_dependencies


if t.TYPE_CHECKING:
    import vllm

    from ._types import DictStrAny
else:
    DictStrAny = dict
    vllm = LazyLoader("vllm", globals(), "vllm")


@attr.frozen(slots=True)
class GenerationInput:
    prompt: str
    """The prompt to be sent to system."""

    llm_config: LLMConfig
    """A mapping of given LLM configuration values for given system."""

    @staticmethod
    def convert_llm_config(data: dict[str, t.Any] | LLMConfig, cls: type[LLMConfig] | None = None) -> LLMConfig:
        if isinstance(data, LLMConfig):
            return data
        elif LazyType(DictStrAny).isinstance(data):
            if cls is None:
                raise ValueError("'cls' must pass if given data is a dictionary.")
            return cls(**data)
        else:
            raise RuntimeError(f"Type {type(data)} is not yet supported.")

    @classmethod
    def for_model(cls, model_name: str, **attrs: t.Any) -> type[GenerationInput]:
        from .models.auto import AutoConfig

        llm_config = AutoConfig.for_model(model_name, **attrs)
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
    supports_embeddings: bool
    supports_hf_agent: bool


@attr.frozen(slots=True)
class EmbeddingsOutput:
    embeddings: t.List[float]
    num_tokens: int


@requires_dependencies("vllm", extra="vllm")
def unmarshal_vllm_outputs(request_output: vllm.RequestOutput) -> DictStrAny:
    return dict(
        request_id=request_output.request_id,
        prompt=request_output.prompt,
        finished=request_output.finished,
        prompt_token_ids=request_output.prompt_token_ids,
        outputs=[
            dict(
                index=it.index,
                text=it.text,
                token_ids=it.token_ids,
                cumulative_logprob=it.cumulative_logprob,
                logprobs=it.logprobs,
                finish_reason=it.finish_reason,
            )
            for it in request_output.outputs
        ],
    )
