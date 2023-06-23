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
import warnings

import attr
import orjson
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import bentoml
import openllm


if t.TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

# The following warnings from bitsandbytes, and probably not that important
# for users to see
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization"
)
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)
warnings.filterwarnings(
    "ignore",
    message=(
        "The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers and GPU quantization"
        " are unavailable."
    ),
)

model = os.environ.get("OPENLLM_MODEL", "{__model_name__}")  # openllm: model name
model_id = os.environ.get("OPENLLM_MODEL_ID", "{__model_id__}")  # openllm: model id
adapter_map = os.environ.get("OPENLLM_ADAPTER_MAP", """{__model_adapter_map__}""")  # openllm: model adapter map

llm_config = openllm.AutoConfig.for_model(model)

runner = openllm.Runner(
    model,
    model_id=model_id,
    llm_config=llm_config,
    bettertransformer=llm_config["env"]["bettertransformer_value"],
    quantize=llm_config["env"]["quantize_value"],
    adapter_map=orjson.loads(adapter_map),
    ensure_available=False,
    init_local=False,
)

svc = bentoml.Service(name=f"llm-{llm_config['start_name']}-service", runners=[runner])


@svc.api(
    input=bentoml.io.JSON.from_sample(sample={"prompt": "", "llm_config": llm_config.model_dump(flatten=True)}),
    output=bentoml.io.JSON.from_sample(sample={"responses": [], "configuration": llm_config.model_dump(flatten=True)}),
    route="/v1/generate",
)
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
    qa_inputs = openllm.GenerationInput.for_model(model)(**input_dict)
    config = qa_inputs.llm_config.model_dump()
    responses = await runner.generate.async_run(qa_inputs.prompt, **config)
    return openllm.GenerationOutput(responses=responses, configuration=config)


@svc.api(
    input=bentoml.io.Text(),
    output=bentoml.io.JSON.from_sample(
        sample={
            "model_id": model_id,
            "timeout": 3600,
            "model_name": llm_config["model_name"],
            "framework": "pt",
            "configuration": "",
        }
    ),
    route="/v1/metadata",
)
def metadata_v1(_: str) -> openllm.MetadataOutput:
    return openllm.MetadataOutput(
        model_id=model_id,
        timeout=llm_config["timeout"],
        model_name=llm_config["model_name"],
        framework=llm_config["env"]["framework_value"],
        configuration=llm_config.model_dump_json().decode(),
    )


@svc.api(
    input=bentoml.io.Text.from_sample(sample="default"),
    output=bentoml.io.JSON.from_sample(sample={"success": True, "error_msg": "some error message"}),
    route="/v1/adapters",
)
async def adapters_v1(adapter_name: str) -> dict[str, bool | str]:
    return await runner.set_adapter.async_run(adapter_name)


@attr.define
class HfAgentInput:
    inputs: str
    parameters: t.Dict[str, t.Any]


async def hf_agent(request: Request) -> Response:
    json_str = await request.body()
    try:
        input_data = openllm.utils.bentoml_cattr.structure(orjson.loads(json_str), HfAgentInput)
    except orjson.JSONDecodeError as err:
        raise openllm.exceptions.OpenLLMException(f"Invalid JSON input received: {err}") from None

    stop = input_data.parameters.pop("stop", "\n")
    try:
        resp = await runner.generate_one.async_run(input_data.inputs, stop, **input_data.parameters)
        return JSONResponse(resp, status_code=200)
    except NotImplementedError:
        return JSONResponse(f"'{model}' is currently not supported with HuggingFace agents.", status_code=500)


hf_app = Starlette(debug=True, routes=[Route("/agent", hf_agent, methods=["POST"])])

svc.mount_asgi_app(hf_app, path="/hf")


async def list_adapter_v1(_: Request) -> Response:
    res = await runner.list_adapter.async_run()
    if res["success"]:
        res["result"] = {k: v.to_dict() for k, v in res["result"].items()}
    return JSONResponse(res, status_code=200)


metadata_app = Starlette(debug=True, routes=[Route("/adapters", list_adapter_v1, methods=["GET"])])
svc.mount_asgi_app(metadata_app, path="/v1")
