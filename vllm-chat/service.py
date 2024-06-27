import functools
import json
import logging
import os
import sys
import uuid
from typing import AsyncGenerator, Literal, Optional

import bentoml
import fastapi
import fastapi.staticfiles
import pydantic
import vllm.entrypoints.openai.api_server as vllm_api_server
import yaml
from annotated_types import Ge, Le
from bento_constants import CONSTANT_YAML
from fastapi.responses import FileResponse
from typing_extensions import Annotated, Literal


class Message(pydantic.BaseModel):
    content: str
    role: Literal["system", "user", "assistant"]


CONSTANTS = yaml.safe_load(CONSTANT_YAML)

ENGINE_CONFIG = CONSTANTS["engine_config"]
SERVICE_CONFIG = CONSTANTS["service_config"]
OVERRIDE_CHAT_TEMPLATE = CONSTANTS.get("chat_template")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


openai_api_app = fastapi.FastAPI()
static_app = fastapi.FastAPI()
ui_app = fastapi.FastAPI()


OPENAI_ENDPOINTS = [
    ["/chat/completions", vllm_api_server.create_chat_completion, ["POST"]],
    ["/completions", vllm_api_server.create_completion, ["POST"]],
    ["/models", vllm_api_server.show_available_models, ["GET"]],
]


class Message(pydantic.BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


for route, endpoint, methods in OPENAI_ENDPOINTS:
    openai_api_app.add_api_route(
        path=route,
        endpoint=endpoint,
        methods=methods,
        include_in_schema=True,
    )


STATIC_DIR = os.path.join(os.path.dirname(__file__), "ui")

ui_app.mount(
    "/static", fastapi.staticfiles.StaticFiles(directory=STATIC_DIR), name="static"
)


@ui_app.get("/")
async def serve_chat_html():
    return FileResponse(os.path.join(STATIC_DIR, "chat.html"))


@ui_app.get("/{full_path:path}")
async def catch_all(full_path: str):
    file_path = os.path.join(STATIC_DIR, full_path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse(os.path.join(STATIC_DIR, "chat.html"))


# special handling for prometheus_client of bentoml
if "prometheus_client" in sys.modules:
    sys.modules.pop("prometheus_client")


@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.mount_asgi_app(ui_app, path="/chat")
@bentoml.service(**SERVICE_CONFIG)
class VLLM:
    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

        ENGINE_ARGS = AsyncEngineArgs(**ENGINE_CONFIG)
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = AutoTokenizer.from_pretrained(ENGINE_CONFIG["model"])
        logger.info(f"VLLM service initialized with model: {ENGINE_CONFIG['model']}")

        if OVERRIDE_CHAT_TEMPLATE:  # use community chat template
            gen_config = _get_gen_config(CONSTANTS["chat_template"])
            chat_template = gen_config["template"]
        else:
            chat_template = None

        model_config = self.engine.engine.get_model_config()

        # inject the engine into the openai serving chat and completion
        vllm_api_server.openai_serving_chat = OpenAIServingChat(
            engine=self.engine,
            served_model_names=[ENGINE_CONFIG["model"]],
            response_role="assistant",
            chat_template=chat_template,
            model_config=model_config,
        )
        vllm_api_server.openai_serving_completion = OpenAIServingCompletion(
            engine=self.engine,
            served_model_names=[ENGINE_CONFIG["model"]],
            model_config=model_config,
            lora_modules=None,
        )

    @bentoml.api(route="/api/generate")
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        model: str = ENGINE_CONFIG["model"],
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
        stop: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        if stop is None:
            stop = []

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

    @bentoml.api(route="/api/chat")
    async def chat(
        self,
        messages: list[Message] = [
            Message(content="what is the meaning of life?", role="user")
        ],
        model: str = ENGINE_CONFIG["model"],
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
        stop: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        light-weight chat API that takes in a list of messages and returns a response
        """
        from vllm import SamplingParams

        try:
            if OVERRIDE_CHAT_TEMPLATE:  # community chat template
                gen_config = _get_gen_config(CONSTANTS["chat_template"])
                if not stop:
                    if gen_config["stop_str"]:
                        stop = [gen_config["stop_str"]]
                    else:
                        stop = []
                system_prompt = gen_config["system_prompt"]
                self.tokenizer.chat_template = gen_config["template"]
            else:
                if not stop:
                    if self.tokenizer.eos_token is not None:
                        stop = [self.tokenizer.eos_token]
                    else:
                        stop = []
                system_prompt = None

            SAMPLING_PARAM = SamplingParams(
                max_tokens=max_tokens,
                stop=stop,
            )
            if system_prompt and messages[0].role != "system":
                messages = [dict(role="system", content=system_prompt)] + messages

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            stream = await self.engine.add_request(
                uuid.uuid4().hex, prompt, SAMPLING_PARAM
            )

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
        except Exception as e:
            logger.error(f"Error in chat API: {e}")
            yield f"Error in chat API: {e}"


@functools.lru_cache(maxsize=1)
def _get_gen_config(community_chat_template: str) -> dict:
    logger.info(f"Load community_chat_template: {community_chat_template}")
    chat_template_path = os.path.join(
        os.path.dirname(__file__), "chat_templates", "chat_templates"
    )
    config_path = os.path.join(
        os.path.dirname(__file__), "chat_templates", "generation_configs"
    )
    with open(os.path.join(config_path, f"{community_chat_template}.json")) as f:
        gen_config = json.load(f)
    chat_template_file = gen_config["chat_template"].split("/")[-1]
    with open(os.path.join(chat_template_path, chat_template_file)) as f:
        chat_template = f.read()
    gen_config["template"] = chat_template.replace("    ", "").replace("\n", "")
    return gen_config
