from __future__ import annotations

import os
import typing as t

import bentoml

import openllm

# NOTE: The below code should not be changed as it will be used by the ast parser
# to generate the service code. This is the current drawback of this approach, but
# good for now. The below make sure that users who use `bentoml serve llm-bento` would
# work.
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
    return {"model_name": llm_config.__openllm_model_name__}
