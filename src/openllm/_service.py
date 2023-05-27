"""
The service definition for running any LLMService.

Note that the line `model = ...` is a special line and should not be modified. This will be handled by openllm 
internally to generate the correct model service when bundling the LLM to a Bento. 
This will ensure that 'bentoml serve llm-bento' will work accordingly.

The generation code lives under ./utils/codegen.py
"""
from __future__ import annotations

import os
import typing as t

import bentoml

import openllm

model = os.environ.get("OPENLLM_MODEL", "{__model_name__}")  # openllm: model name

llm_config = openllm.AutoConfig.for_model(model)
runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(name=f"llm-{llm_config.__openllm_start_name__}-service", runners=[runner])


@svc.api(
    input=bentoml.io.JSON(pydantic_model=openllm.GenerationInput.for_model(model)),
    output=bentoml.io.JSON(pydantic_model=openllm.GenerationOutput),
    route="/v1/generate",
)
async def generate_v1(qa: openllm.GenerationInput) -> openllm.GenerationOutput:
    config = llm_config.with_options(__llm_config__=qa.llm_config).model_dump()
    responses = await runner.generate.async_run(qa.prompt, **config)
    return openllm.GenerationOutput(responses=responses, configuration=config)


@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON(), route="/v1/metadata")
def metadata_v1(_: str) -> dict[str, t.Any]:
    return {
        "model_name": llm_config.__openllm_model_name__,
        "framework": openllm.utils.get_framework_env(llm_config.__openllm_model_name__),
    }
