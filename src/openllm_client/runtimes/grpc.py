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

import bentoml

# NOTE: Add this import here to register the IO Descriptor.
from openllm import Prompt as Prompt

from .base import BaseClient

logger = logging.getLogger(__name__)

import asyncio

import grpc_health.v1.health_pb2 as health_pb2


class GrpcClient(BaseClient):
    __: bentoml.client.GrpcClient | None = None
    _model_setup: bool = False

    def __init__(self, address: str, timeout: int = 30):
        self._timeout = timeout
        self._address = address
        self._host, self._port = address.split(":")

    @property
    def _cached(self) -> bentoml.client.GrpcClient:
        if self.__ is None:
            bentoml.client.GrpcClient.wait_until_server_ready(self._host, int(self._port), timeout=self._timeout)
            self.__ = bentoml.client.GrpcClient.from_url(self._address)
        return self.__

    def setup(self, **llm_config_args: t.Any):
        if self._model_setup:
            logger.warning(
                "LLM is already setup. Send runtime parameter per each request (%s) instead. Ignoring.", llm_config_args
            )
            return

        try:
            self._cached.set_default_config(llm_config_args)
            self._model_setup = True
        except bentoml.exceptions.BentoMLException as e:
            logger.error("Failed to setup LLM.")
            logger.error(e)
            raise

    def health(self) -> health_pb2.HealthCheckResponse:
        return asyncio.run(self._cached.health("bentoml.grpc.v1.BentoService"))
