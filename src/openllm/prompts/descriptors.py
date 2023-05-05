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

Example:
    >>> from openllm.prompts import Prompt
    >>> prompt = Prompt("Use the following as context: {context}\nQuestion: {question}\nAnswer:")

It also adds a ``from_http`` and ``to_http`` method which wraps around ``from_http_request`` and ``to_http_response``.

TODO: 
- Add support for ``langchain`` PromptTemplate under ``template``.

Prompt can also be set via ``OPENLLM_PROMPT_TEMPLATE`` environment variable.
"""
from __future__ import annotations

import importlib
import inspect
import io
import logging
import os
import typing as t

import bentoml
import orjson
import pydantic
import pydantic.schema
import yaml
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


class Prompt(bentoml.io.IODescriptor[pydantic.BaseModel], descriptor_id=f"{__name__}.Prompt", proto_fields=("json",)):
    _mime_type = "application/json"
    _config_class = openllm.LLMConfig

    # TODO: support langchain PromptTemplate under template
    prompt_template: openllm.schema.PromptTemplate

    input_schema = openllm.schema.PromptInput
    output_schema = openllm.schema.PromptOutput

    def __setattr__(self, attr_name: str, value: t.Any) -> None:
        if attr_name in ("input_schema", "output_schema"):
            raise openllm.exceptions.OpenLLMException(f"{attr_name} is frozen as custom schema is not yet supported.")
        super().__setattr__(attr_name, value)

    def __special__(self):
        frame = inspect.currentframe()
        f_locals = frame.f_back.f_back.f_locals
        _config_key = f_locals.get("__use_config__", None)
        if _config_key is not None:
            try:
                self._config_class = openllm.CONFIG_MAPPING[_config_key]
            except KeyError:
                logger.debug(
                    "Invalid config key '%s'. Please choose from: %s",
                    _config_key,
                    ", ".join(openllm.CONFIG_MAPPING.keys()),
                )
        else:
            self._config_class = openllm.LLMConfig

    def __init__(self, template: str | None = None, config_class: type[openllm.LLMConfig] | None = None):
        # The behaviour is as follow:
        # 1. If template is passed, then we will use it as template. We will also ask user to specify
        #   the config class to be used for this prompt.
        # 2. If only config_class is passed, then we will use the default prompt inferred from this config class.
        # 3. EXPERIMENTAL: If neither are passed, we will try to inspect for '__use_config__' in the service definition
        #   and use that as the config class. This is to ensure that we can dynamically pass config class per request.

        # We will always respect environment variable over the default value, even with `__use_config__`
        # is available in service definition.
        template_from_env = os.environ.get("OPENLLM_PROMPT_TEMPLATE", None)

        # 1. Processing templates
        #   We will also accept prompt passing from environment variables.
        if template is not None:
            if config_class is None:
                # If config class is not passed, but special value is passed, then we will check for special value.
                self.__special__()
                if self._config_class is openllm.LLMConfig:
                    raise RuntimeError(
                        "Please specify config_class in addition to template to pass LLM configuration per request."
                    )
                # __use_config__ is available, then we can continue
                template = template_from_env if template_from_env else template
                self.prompt_template = openllm.schema.PromptTemplate.from_template(template)
            else:
                # If config class is passed, then we will use it.
                self._config_class = config_class
                self.prompt_template = openllm.schema.PromptTemplate.from_template(template)
        else:
            # 2. When template is not passed, but config class is,
            # we can then determine the default template in
            # adjacent to the environment variable.
            if config_class is not None:
                if not template_from_env and not hasattr(
                    importlib.import_module(config_class.__module__), "DEFAULT_PROMPT_TEMPLATE"
                ):
                    raise RuntimeError(
                        "Exhausted all available options to determine default templates."
                        " Make sure to pass the template either with the config_class"
                        " in 'Prompt' or through 'OPENLLM_PROMPT_TEMPLATE' environment variable."
                    )
                template = (
                    template_from_env
                    if template_from_env
                    else getattr(importlib.import_module(config_class.__module__), "DEFAULT_PROMPT_TEMPLATE")
                )
                self.prompt_template = openllm.schema.PromptTemplate.from_template(template)
                self._config_class = config_class
            else:
                # 3. config_class and template is all None, check for special value.
                if self._config_class is openllm.LLMConfig:
                    self.__special__()
                if self._config_class is openllm.LLMConfig:
                    raise RuntimeError(
                        "Failed to auto determine config class and template. "
                        "Make sure to specify them in the 'Prompt' constructor."
                    )
                template = (
                    template_from_env
                    if template_from_env
                    else getattr(importlib.import_module(self._config_class.__module__), "DEFAULT_PROMPT_TEMPLATE")
                )
                self.prompt_template = openllm.schema.PromptTemplate.from_template(template)

    @property
    def template(self) -> str:
        assert self.prompt_template, "'Prompt' is not initialized correctly."
        return self.prompt_template.template

    @property
    def input_variables(self) -> t.Sequence[str]:
        assert self.prompt_template, "'Prompt' is not initialized correctly."
        return self.prompt_template.input_variables

    async def from_http(self, request: Request) -> openllm.schema.PromptInput:
        """
        Parse the JSON body from HTTP request and return the PromptInput.

        The body can have two format:
        1. {"context": "...", "question": "..."}: dict[str, str] of the key that coresponding to self.input_variables
        2. {"different_var": "...", "di2": "...", 'prompt': "... {different_var} ... {di2} ..."}: This dictionary contains a different few-shot prompts, hence we will use that to parse the remaining variable.
        3. {"different_var": "...", "di2": "...", 'prompt': "... {different_var} ... {di2} ...", "temperature": 0.8, "top_p": 0.2}: 1 | 2 plus the model configuration per requests.
        """
        json_str = await request.body()
        try:
            parsed = orjson.loads(json_str)
        except orjson.JSONDecodeError as e:
            raise bentoml.exceptions.InvalidArgument(f"Invalid JSON: {e}")

        _prompt = self.prompt_template

        if "prompt" in parsed:
            #  In this branch, user pass in a full prompt.
            #  We need to parse it and extract the input variables.
            prompt = parsed.pop("prompt")
            _prompt = openllm.schema.PromptTemplate.from_template(prompt)

        llm_config_ = {k: v for k, v in parsed.items() if k in self._config_class.__fields__}

        for k in llm_config_:
            parsed.pop(k)

        return self.input_schema(prompt=_prompt.to_str(**parsed), llm_config=llm_config_)

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

        _prompt = self.prompt_template

        if "prompt" in parsed:
            #  In this branch, user pass in a full prompt.
            #  We need to parse it and extract the input variables.
            prompt = parsed.pop("prompt")
            _prompt = openllm.schema.PromptTemplate.from_template(prompt)

        llm_config_ = {k: v for k, v in parsed.items() if k in self._config_class.__fields__}

        for k in llm_config_:
            parsed.pop(k)

        return self.input_schema(prompt=_prompt.to_str(**parsed), llm_config=llm_config_)

    async def to_http(self, obj: pydantic.BaseModel | dict[str, t.Any], ctx: bentoml.Context | None = None):
        if isinstance(obj, dict):
            try:
                obj = self.output_schema(**obj)
            except pydantic.ValidationError as e:
                raise bentoml.exceptions.BadInput(f"Invalid output: {e}")
        elif not isinstance(obj, openllm.schema.PromptOutput):
            raise bentoml.exceptions.BadInput(f"Expected output of type {openllm.schema.PromptOutput}, got {type(obj)}")

        json_str = orjson.dumps(obj.dict())
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

    async def to_proto(self, obj: pydantic.BaseModel | dict[str, t.Any]) -> struct_pb2.Value:
        if isinstance(obj, dict):
            try:
                obj = self.output_schema(**obj)
            except pydantic.ValidationError as e:
                raise bentoml.exceptions.BadInput(f"Invalid output: {e}")
        elif not isinstance(obj, openllm.schema.PromptOutput):
            raise bentoml.exceptions.BadInput(f"Expected output of type {openllm.schema.PromptOutput}, got {type(obj)}")

        json_ = obj.dict()

        msg = struct_pb2.Value()
        return parse_dict_to_proto(json_, msg)

    # NOTE: OpenAPI specification for Prompt to be a bentoml.io.IODescriptor
    def input_type(self) -> type[pydantic.BaseModel]:
        return pydantic.BaseModel

    def to_spec(self) -> dict[str, t.Any]:
        return {
            "id": self.descriptor_id,
            "args": {
                "template": self.prompt_template.template,
                "config_class_module": self._config_class.__module__,
                "config_class_name": self._config_class.__name__,
            },
        }

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        if "args" not in spec:
            raise bentoml.exceptions.InvalidArgument(f"Missing args key in JSON spec: {spec}")
        klass = getattr(importlib.import_module(spec["args"]["config_class_module"]), spec["args"]["config_class_name"])
        return cls(config_class=klass, template=spec["args"]["template"])

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
        return {}  # TODO:

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
                    "example": self.openapi_output_example(),
                }
            },
            "x-bentoml-io-descriptor": self.to_spec(),
        }

    # NOTE: The below override default the loose contract from bentoml.io.IODescriptor
    @classmethod
    def from_sample(cls, _: t.Any) -> Prompt:
        raise NotImplementedError("Prompt does not support 'from_sample'.")

    def _from_sample(self, sample: t.Any) -> pydantic.BaseModel:
        return sample

    def openapi_schema(self) -> t.Any:
        # NOTE: not used
        return {"type": "object"}

    async def from_http_request(self, request: Request) -> openllm.schema.PromptInput:
        return await self.from_http(request)

    async def to_http_response(self, obj: pydantic.BaseModel, ctx: bentoml.Context | None = None):
        return await self.to_http(obj, ctx)
