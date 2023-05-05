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

import logging
import typing as t
from urllib.parse import urlparse

import bentoml

import openllm

from .base import BaseClient

logger = logging.getLogger(__name__)


class HTTPClient(BaseClient):
    __: bentoml.client.HTTPClient | None = None

    def __init__(self, address: str, timeout: int = 30):
        address = address if "://" in address else "http://" + address
        self._timeout = timeout
        self._address = address
        self._host, self._port = urlparse(address).netloc.split(":")

    @property
    def _cached(self) -> bentoml.client.HTTPClient:
        if self.__ is None:
            bentoml.client.HTTPClient.wait_until_server_ready(self._host, int(self._port), timeout=self._timeout)
            self.__ = bentoml.client.HTTPClient.from_url(self._address)
        return self.__

    def health(self) -> t.Any:
        return self._cached.health()

    def query(self, prompt_template: str | openllm.PromptTemplate | None = None, **attrs: t.Any) -> str:
        return_raw_response = attrs.pop("return_raw_response", False)
        model_name = self._cached.model_name()
        if prompt_template is None:
            # return the default prompt
            prompt_template = openllm.PromptTemplate.from_default(model_name)
        elif isinstance(prompt_template, str):
            prompt_template = openllm.PromptTemplate.from_template(prompt_template)
        variables = {k: v for k, v in attrs.items() if k in prompt_template.input_variables}
        config = openllm.AutoConfig.for_model(model_name).with_options(
            **{k: v for k, v in attrs.items() if k not in variables}
        )
        r = openllm.schema.GenerateOutput(
            **self._cached.generate(
                openllm.schema.GenerateInput(prompt=prompt_template.to_str(**variables), llm_config=config.dict())
            )
        )
        if return_raw_response:
            return r.dict()

        return prompt_template.to_str(**variables) + "".join(r.responses)

    def chat(
        self,
        prompt: str,
        context: str,
        prompt_template: str | openllm.PromptTemplate | None = None,
        **llm_config: t.Any,
    ):
        ...
