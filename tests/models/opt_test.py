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
import typing as t

import pytest

import openllm


if t.TYPE_CHECKING:
    import contextlib

    from .conftest import HandleProtocol
    from .conftest import ResponseComparator
    from .conftest import _Handle


model = "opt"
model_id = "facebook/opt-125m"


@pytest.fixture(scope="module")
def opt_125m_handle(
    handler: HandleProtocol,
    deployment_mode: t.Literal["container", "local"],
    clean_context: contextlib.ExitStack,
):
    with openllm.testing.prepare(
        model, model_id=model_id, deployment_mode=deployment_mode, clean_context=clean_context
    ) as image_tag:
        with handler(model=model, model_id=model_id, image_tag=image_tag) as handle:
            yield handle


@pytest.fixture(scope="module")
async def opt_125m(opt_125m_handle: _Handle):
    await opt_125m_handle.health(240)
    return opt_125m_handle.client


@pytest.mark.asyncio()
async def test_opt_125m(opt_125m: t.Awaitable[openllm.client.AsyncHTTPClient], response_snapshot: ResponseComparator):
    client = await opt_125m
    response = await client.query("What is Deep learning?", max_new_tokens=20, return_attrs=True)

    assert response.configuration["generation_config"]["max_new_tokens"] == 20
    assert response == response_snapshot
