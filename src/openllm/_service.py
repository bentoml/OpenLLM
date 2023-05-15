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

import os
import typing as t

import bentoml

import openllm

model = os.environ.get("OPENLLM_MODEL", None)

if model is None:
    raise RuntimeError("OPENLLM_MODEL environment variable must be set.")

llm_config = openllm.AutoConfig.for_model(model)
runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(name=f"llm-{llm_config.__openllm_start_name__}-service", runners=[runner])


@svc.api(
    input=bentoml.io.JSON(pydantic_model=openllm.GenerationInput),
    output=bentoml.io.JSON(pydantic_model=openllm.GenerationOutput),
    route="/v1/generate",
)
async def generate_v1(qa: openllm.GenerationInput) -> openllm.GenerationOutput:
    """Generate a response from the given prompt."""
    config = llm_config.with_options(**qa.llm_config).dict()
    responses = await svc.runners[0].generate.async_run(qa.prompt, **config)
    return openllm.GenerationOutput(responses=responses, configuration=config)


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON(), route="/v1/metadata")
def metadata_v1(_: str) -> dict[str, t.Any]:
    """Get the metadata for this service."""
    return {"model_name": llm_config.__openllm_model_name__}
