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
The service definition for running any LLMService.

Note that the line `model = ...` is a special line and should not be modified. This will be handled by openllm
internally to generate the correct model service when bundling the LLM to a Bento.
This will ensure that 'bentoml serve llm-bento' will work accordingly.

The generation code lives under utils/codegen.py
"""
from __future__ import annotations

import os
import typing as t

import bentoml

import openllm

model = os.environ.get("OPENLLM_MODEL", "{__model_name__}")  # openllm: model name
model_id = os.environ.get("OPENLLM_MODEL_ID", "{__model_id__}")  # openllm: model id

llm_config = openllm.AutoConfig.for_model(model)
runner = openllm.Runner(model, model_id=model_id, llm_config=llm_config)

svc = bentoml.Service(name=f"llm-{llm_config.__openllm_start_name__}-service", runners=[runner])


@svc.on_deployment
def ensure_exists():
    # NOTE: We need to initialize llm here first to check if the model is already downloaded to
    # avoid deadlock before the subprocess forking.
    runner.llm.ensure_pretrained_exists()


@svc.api(
    input=bentoml.io.JSON.from_sample(sample={"prompt": "", "llm_config": llm_config.model_dump()}),
    output=bentoml.io.JSON.from_sample(sample={"responses": [], "configuration": llm_config.model_dump()}),
    route="/v1/generate",
)
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
    qa_inputs = openllm.GenerationInput.for_model(model)(**input_dict)
    config = qa_inputs.llm_config.model_dump()
    responses = await runner.generate.async_run(qa_inputs.prompt, **config)
    return openllm.GenerationOutput(responses=responses, configuration=config)


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON(), route="/v1/metadata")
def metadata_v1(_: str) -> dict[str, t.Any]:
    return {
        "timeout": llm_config.__openllm_timeout__,
        "model_name": llm_config.__openllm_model_name__,
        "framework": llm_config.__openllm_env__.get_framework_env(),
    }
