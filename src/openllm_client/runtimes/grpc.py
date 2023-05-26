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

import asyncio
import logging
import typing as t

import openllm

from .base import BaseClient

if t.TYPE_CHECKING:
    import grpc_health.v1.health_pb2 as health_pb2

logger = logging.getLogger(__name__)


class GrpcClient(BaseClient, client_type="grpc"):
    def __init__(self, address: str, timeout: int = 30):
        self._host, self._port = address.split(":")
        super().__init__(address, timeout)

    def health(self) -> health_pb2.HealthCheckResponse:
        return asyncio.run(self._cached.health("bentoml.grpc.v1.BentoService"))

    def query(self, prompt: str, **attrs: t.Any) -> dict[str, t.Any] | list[t.Any]:
        return_raw_response = attrs.pop("return_raw_response", False)
        r = openllm.GenerationOutput(
            **self.call(
                "generate",
                openllm.GenerationInput(
                    prompt=prompt,
                    llm_config=self.config.with_options(**attrs),
                ),
            )
        )
        if return_raw_response:
            return r.model_dump()

        return r.responses
