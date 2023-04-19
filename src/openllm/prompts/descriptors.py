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
Descriptor definition for OpenLLM. ``Prompt`` in hindsight extends the functionality of ``bentoml.io.JSON``.

However, it uses ``orjson`` instead of ``json`` for faster serialization and deserialization.

One distinct difference is that ``Prompt`` can only be initialized with ``Prompt.from_template``.

Example:
    >>> from openllm.prompts import Prompt
    >>> prompt = Prompt.from_template("Use the following as context: {context}!")
    >>> prompt = Prompt("Use the following as context: {context}\nQuestion: {question}\nAnswer:")

It also adds a ``from_http`` and ``to_http`` method which wraps around ``from_http_request`` and ``to_http_response``.

TODO: 
- Add support for ``langchain`` PromptTemplate under ``template``.

Prompt can also be set via ``OPENLLM_PROMPT_TEMPLATE`` environment variable.
"""
from __future__ import annotations

import logging
import os
import typing as t

import bentoml
import orjson
import pydantic
import pydantic.schema
from bentoml._internal.io_descriptors.json import parse_dict_to_proto
from bentoml._internal.utils.http import set_cookies
from starlette.requests import Request
from starlette.responses import Response

import openllm

if t.TYPE_CHECKING:
    from google.protobuf import struct_pb2

    from openllm.types import OpenAPIResponse
else:
    struct_pb2 = openllm.utils.LazyLoader("struct_pb2", globals(), "google.protobuf.struct_pb2")

logger = logging.getLogger(__name__)


class Prompt(
    bentoml.io.IODescriptor[pydantic.BaseModel],
    descriptor_id="openllm.prompts.descriptors.Prompt",
    proto_fields=("json",),
):
    _mime_type = "application/json"

    # TODO: support langchain PromptTemplate under template
    prompt_template: openllm.schema.PromptTemplate

    input_schema = openllm.schema.PromptInput
    output_schema = openllm.schema.PromptOutput

    def __setattr__(self, attr_name: str, value: t.Any) -> None:
        if attr_name in ("input_schema", "output_schema"):
            raise openllm.exceptions.OpenLLMException(f"{attr_name} is frozen as custom schema is not yet supported.")
        super().__setattr__(attr_name, value)

    def __init__(
        self,
        template: openllm.schema.PromptTemplate | openllm.schema.PromptInput | str | None = None,
        default: str | None = None,
    ) -> None:
        if default:
            default = openllm.utils.kebab_to_snake_case(default)
        template_from_env = os.environ.get("OPENLLM_PROMPT_TEMPLATE", None)
        if template is None and template_from_env is None:
            if default:
                if default not in openllm.CONFIG_MAPPING.keys():
                    raise ValueError(
                        "Invalid default prompt template. Please choose from: "
                        + ", ".join(openllm.CONFIG_MAPPING.keys())
                    )
                self.prompt_template = openllm.schema.PromptTemplate.from_default(default)
                return
            else:
                raise ValueError(
                    "Prompt template is not set. Please set it via 'OPENLLM_PROMPT_TEMPLATE' environment variable or pass it to 'Prompt'."
                )

        # This logic ensure that we will always respect the template from env if set, even if `template` is passed
        template = template_from_env if template_from_env is not None else template

        if isinstance(template, str):
            self.prompt_template = openllm.schema.PromptTemplate.from_template(template)
        elif isinstance(template, openllm.schema.PromptInput):
            self.prompt_template = openllm.schema.PromptTemplate.from_template(template.prompt)
        elif isinstance(template, openllm.schema.PromptTemplate):
            self.prompt_template = template
        else:
            raise openllm.exceptions.OpenLLMException(
                f"Invalid sample type: {type(template)}. Sample must be one of: {openllm.schema.PromptInput}, {openllm.schema.PromptTemplate} or str."
            )

    @classmethod
    def from_template(
        cls,
        template: openllm.schema.PromptTemplate | openllm.schema.PromptInput | str | None = None,
        default: str | None = None,
    ) -> Prompt:
        return cls(template=template, default=default)

    @property
    def template(self) -> str:
        assert self.prompt_template, "Make sure to initialize Prompt with 'from_template' first."
        return self.prompt_template.template

    @property
    def input_variables(self) -> t.Sequence[str]:
        assert self.prompt_template, "Make sure to initialize Prompt with 'from_template' first."
        return self.prompt_template.input_variables

    async def from_http(self, request: Request) -> openllm.schema.PromptInput:
        json_str = await request.body()
        try:
            parsed = orjson.loads(json_str)
        except orjson.JSONDecodeError as e:
            raise bentoml.exceptions.InvalidArgument(f"Invalid JSON: {e}")

        if "prompt" in parsed:
            #  In this branch, user pass in a full prompt.
            #  We need to parse it and extract the input variables.
            prompt = parsed.pop("prompt")
            prompt_template = openllm.schema.PromptTemplate.from_template(prompt)
            return self.input_schema(prompt=prompt_template.to_str(**parsed), inputs=parsed)
        # In this branch, users only pass in the input variables for set prompt
        # (either default or set via environment variables)
        return self.input_schema(prompt=self.prompt_template.to_str(**parsed), inputs=parsed)

    async def from_proto(self, field: struct_pb2.Value | bytes) -> openllm.schema.PromptInput:
        from google.protobuf.json_format import MessageToDict

        if isinstance(field, bytes):
            content = field
            try:
                parsed = orjson.loads(content)
            except orjson.JSONDecodeError as e:
                raise bentoml.exceptions.BadInput(f"Invalid JSON: {e}")
        else:
            assert isinstance(field, struct_pb2.Value)
            parsed = MessageToDict(field, preserving_proto_field_name=True)

        if "prompt" in parsed:
            #  In this branch, user pass in a full prompt.
            #  We need to parse it and extract the input variables.
            prompt = parsed.pop("prompt")
            prompt_template = openllm.schema.PromptTemplate.from_template(prompt)
            return self.input_schema(prompt=prompt_template.to_str(**parsed), inputs=parsed)
        # In this branch, users only pass in the input variables for set prompt
        # (either default or set via environment variables)
        return self.input_schema(prompt=self.prompt_template.to_str(**parsed), inputs=parsed)

    async def to_http(self, obj: pydantic.BaseModel, ctx: bentoml.Context | None = None):
        if not isinstance(obj, openllm.schema.PromptOutput):
            raise bentoml.exceptions.BadInput(f"Expected output of type {openllm.schema.PromptOutput}, got {type(obj)}")
        logger.debug("Converting '%s' to '%s'", obj, self.output_schema)

        json_str = orjson.dumps(self.output_schema(**obj.dict()).dict())
        if ctx is not None:
            res = Response(
                json_str,
                media_type=self._mime_type,
                headers=ctx.response.metadata,
                status_code=ctx.response.status_code,
            )
            set_cookies(res, ctx.response.cookies)
        else:
            res = Response(json_str, media_type=self._mime_type)

        return res

    async def to_proto(self, obj: pydantic.BaseModel) -> struct_pb2.Value:
        if not isinstance(obj, openllm.schema.PromptOutput):
            raise bentoml.exceptions.BadInput(f"Expected output of type {openllm.schema.PromptOutput}, got {type(obj)}")
        logger.debug("Converting '%s' to '%s'", obj, self.output_schema)

        json_ = self.output_schema(**obj.dict()).dict()

        msg = struct_pb2.Value()
        return parse_dict_to_proto(json_, msg)

    # NOTE: OpenAPI specification for Prompt to be a bentoml.io.IODescriptor
    def input_type(self) -> type[pydantic.BaseModel]:
        return pydantic.BaseModel

    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": self.descriptor_id,
            "args": {"template": self.prompt_template.template},
        }

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        if "args" not in spec:
            raise bentoml.exceptions.InvalidArgument(f"Missing args key in JSON spec: {spec}")
        return cls.from_template(**spec["args"])

    def openapi_schema(self) -> t.Any:
        # NOTE: not used
        return {"type": "object"}

    @property
    def _model_name_map(self) -> dict[pydantic.schema.TypeModelOrEnum, str]:
        return pydantic.schema.get_model_name_map(
            pydantic.schema.get_flat_models_from_models({self.input_schema, self.output_schema})
        )

    def openapi_input_schema(self) -> dict[str, t.Any]:
        return pydantic.schema.model_process_schema(
            self.input_schema, model_name_map=self._model_name_map, ref_prefix="#/components/schemas/"
        )[0]

    def openapi_output_schema(self) -> dict[str, t.Any]:
        return pydantic.schema.model_process_schema(
            self.output_schema, model_name_map=self._model_name_map, ref_prefix="#/components/schemas/"
        )[0]

    def openapi_components(self) -> dict[str, t.Any] | None:
        # TODO: Support custom input and output schema in BentoML OpenAPI components parsing.
        return

    def openapi_example(self) -> dict[str, t.Any]:
        return {k: "" for k in self.input_variables}

    def openapi_input_example(self) -> dict[str, t.Any]:
        return self.openapi_example()

    def openapi_output_example(self) -> dict[str, t.Any]:
        return {}

    def openapi_request_body(self) -> dict[str, t.Any]:
        return {
            "content": {
                self._mime_type: {
                    "schema": self.openapi_input_schema(),
                    "example": self.openapi_input_example(),
                }
            },
            "required": True,
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    def openapi_responses(self) -> OpenAPIResponse:
        return {
            "description": "Prompt received successfully!",
            "content": {
                self._mime_type: {
                    "schema": self.openapi_output_schema(),
                    "example": self.openapi_output_example(),  # TODO: Support output example
                }
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    # NOTE: The below override default the loose contract from bentoml.io.IODescriptor
    @classmethod
    def from_sample(cls, sample: openllm.schema.PromptInput | str) -> Prompt:
        return cls.from_template(sample)

    def _from_sample(self, sample: t.Any) -> pydantic.BaseModel:
        return sample

    async def from_http_request(self, request: Request) -> openllm.schema.PromptInput:
        return await self.from_http(request)

    async def to_http_response(self, obj: pydantic.BaseModel, ctx: bentoml.Context | None = None):
        return await self.to_http(obj, ctx)
