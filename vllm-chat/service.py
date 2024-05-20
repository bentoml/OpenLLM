import uuid
import json
import os
from typing import AsyncGenerator
from typing_extensions import Annotated

import bentoml
from annotated_types import Ge, Le
from bentovllm_openai.utils import openai_endpoints
import yaml
from bento_constants import CONSTANT_YAML


CONSTANTS = yaml.safe_load(CONSTANT_YAML)

ENGINE_CONFIG = CONSTANTS["engine_config"]
CHAT_TEMPLATE = CONSTANTS["chat_template"]
SERVICE_CONFIG = CONSTANTS["service_config"]


CHAT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "chat_templates", "chat_templates"
)
GEN_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "chat_templates", "generation_configs"
)


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
        stop: list[str] = [],
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            stop=stop,
        )
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)

    @bentoml.api
    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
    ) -> AsyncGenerator[str, None]:
        """
        light-weight chat API that takes in a list of messages and returns a response
        """

        from vllm import SamplingParams
        import jinja2

        JINJA_ENV = jinja2.Environment(
            loader=jinja2.FileSystemLoader(CHAT_TEMPLATE_PATH)
        )

        with open(os.path.join(GEN_CONFIG_PATH, f"{CHAT_TEMPLATE}.json")) as f:
            gen_config = json.load(f)
        chat_template_file = gen_config["chat_template"].split("/")[-1]
        template = JINJA_ENV.get_template(chat_template_file)

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=gen_config["stop_token_ids"],
            stop=(
                [gen_config["stop_str"]] if gen_config["stop_str"] is not None else []
            ),
        )
        if gen_config["system_prompt"] and messages[0].get("role") != "system":
            messages = [
                dict(role="system", content=gen_config["system_prompt"])
            ] + messages
        prompt = template.render(messages=messages, add_generation_prompt=True)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        strip_flag = True
        async for request_output in stream:
            text = request_output.outputs[0].text
            assistant_message = text[cursor:]
            if not strip_flag:  # strip the leading whitespace
                yield assistant_message
            elif assistant_message.strip():
                strip_flag = False
                yield assistant_message.lstrip()
            cursor = len(text)
