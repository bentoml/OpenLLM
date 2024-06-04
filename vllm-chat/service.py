import uuid
import json
import os
from typing import AsyncGenerator, Union
from typing_extensions import Annotated
import functools

import bentoml
from annotated_types import Ge, Le
from bentovllm_openai.utils import openai_endpoints
import yaml
from bento_constants import CONSTANT_YAML


CONSTANTS = yaml.safe_load(CONSTANT_YAML)

ENGINE_CONFIG = CONSTANTS["engine_config"]
SERVICE_CONFIG = CONSTANTS["service_config"]


@functools.lru_cache(maxsize=1)
def _get_gen_config(community_chat_template: str) -> dict:
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


@openai_endpoints(
    served_model_names=[ENGINE_CONFIG["model"]],
    chat_template_model_id=ENGINE_CONFIG["model"],
)
@bentoml.service(**SERVICE_CONFIG)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer

        ENGINE_ARGS = AsyncEngineArgs(**ENGINE_CONFIG)
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = AutoTokenizer.from_pretrained(ENGINE_CONFIG["model"])

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
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        model: str = "",
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
        stop: Union[list[str], str, None] = None,
        stop_token_ids: Union[list[int], None] = None,
    ) -> AsyncGenerator[str, None]:
        """
        light-weight chat API that takes in a list of messages and returns a response
        """
        from vllm import SamplingParams

        if CONSTANTS.get("chat_template"):  # community chat template
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

        # normalize inputs
        if stop_token_ids is None:
            stop_token_ids = []

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            stop=stop,
        )
        if system_prompt and messages[0].get("role") != "system":
            messages = [dict(role="system", content=system_prompt)] + messages

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
