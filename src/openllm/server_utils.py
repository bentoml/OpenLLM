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
Server utilities for OpenLLM. This extends bentoml.Server.

It independently manage processes and threads for runners and servers separately. 
This is an experimental feature and can also be merged to upstream BentoML.
"""
from __future__ import annotations

import logging
import os
import subprocess
import typing as t
from io import StringIO

import bentoml

import openllm

logger = logging.getLogger(__name__)


def _start(
    model_name: str,
    framework: t.Literal["flax", "tf", "pt"] | None = None,
    _serve_grpc: bool = False,
    **attrs: t.Any,
):
    # NOTE: We need the below imports so that the client can use the custom IO Descriptor.
    from openllm.prompts import Prompt as Prompt

    if framework is not None:
        os.environ[openllm.utils.FRAMEWORK_ENV_VAR(model_name)] = framework

    openllm.Config.for_model(model_name)

    server_args = server_args or {}
    server_args.update(
        {
            "working_dir": openllm.utils.get_working_dir(model_name),
            "bento": f'service_{model_name.replace("-", "_")}:svc',
        }
    )
    # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
    development = server_args.pop("development")
    server_args.setdefault("production", not development)
    server = getattr(bentoml, "HTTPServer" if not serve_grpc else "GrpcServer")(**server_args)
    server.timeout = 90

    server.start(env=start_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert server.process is not None
    client = server.get_client()
    llm_config_args = llm_config_args or {}
    if llm_config_args:
        res = client.update_llm_config(llm_config_args)
        assert res

    logger.info("Server for running '%s' can now be accessed at %s", model_name, client.server_url)
    # TODO: Add generated instruction for using client in JS, Python and Go here.

    def log_output(pipe: t.TextIO):
        for line in iter(pipe.readline, b""):  # b'\n'-separated lines
            logger.info(line)

    try:
        stdout, _ = server.process.communicate()
        log_output(StringIO(stdout))
    except Exception as err:
        logger.error("Exception occured while running '%s':\n", model_name)
        logger.error(err)
        raise
