import uuid
from typing import AsyncGenerator
from typing_extensions import Annotated

import bentoml
from annotated_types import Ge, Le
from bentovllm_openai.utils import openai_endpoints
import yaml


CONFIG = yaml.safe_load(open("config.yaml"))

ENGINE_CONFIG = CONFIG["engine_config"]
PROMPT_TEMPLATE = CONFIG["prompt"]
SERVICE_CONFIG = CONFIG["service_config"]


@openai_endpoints(served_model_names=[ENGINE_CONFIG["model"]])
@bentoml.service(**SERVICE_CONFIG)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(**ENGINE_CONFIG)
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
        prompt = (PROMPT_TEMPLATE["head"] or "") + PROMPT_TEMPLATE["body"].format(
            user_prompt=prompt
        )
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
