import uuid
import os
from typing import AsyncGenerator
from typing_extensions import Annotated

import bentoml
from annotated_types import Ge, Le
from bentovllm_openai.utils import openai_endpoints


PROMPT_TEMPLATE_MAP = {
    "llama2:7b-chat": """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{user_prompt} [/INST] """,
    "mistral:7b-instruct": """<s>[INST]
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{user_prompt} [/INST] """,
    "mixtral:8x7b-instruct": """<s>[INST]
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{user_prompt} [/INST] """,
}

MODEL_ID_MAP = {
    "llama2:7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "mistral:7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral:8x7b-instruct": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
}

SERVICE_CONFIG_MAP = {
    "llama2:7b-chat": {
        "name": "llama2",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_type": "nvidia-tesla-t4",
        },
    },
    "mistral:7b-instruct": {
        "name": "mistral",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_type": "nvidia-l4",
        },
    },
    "mixtral:8x7b-instruct": {
        "name": "mixtral",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_type": "nvidia-a100-80gb",
        },
    },
}

MODEL_CONFIG_MAP = {
    "llama2:7b-chat": {
        "max_model_len": 1024,
    },
    "mistral:7b-instruct": {
        "max_model_len": 1024,
        "dtype": "half",
    },
    "mixtral:8x7b-instruct": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "quantization": "gptq",
        "dtype": "half",
    },
}

MODEL_ALIAS_MAP = {
    "llama2": "llama2:7b-chat",
    "llama2:7b": "llama2:7b-chat",
    "llama2:7b-chat": "llama2:7b-chat",
    "mistral": "mistral:7b-instruct",
    "mistral:7b": "mistral:7b-instruct",
    "mistral:7b-instruct": "mistral:7b-instruct",
    "mixtral": "mixtral:8x7b-instruct",
    "mixtral:8x7b": "mixtral:8x7b-instruct",
    "mixtral:8x7b-instruct": "mixtral:8x7b-instruct",
}


MODEL_ALIAS = os.environ["CLLAMA_MODEL"]
MODEL = MODEL_ALIAS_MAP[MODEL_ALIAS]
PROMPT_TEMPLATE = PROMPT_TEMPLATE_MAP[MODEL]
MODEL_ID = MODEL_ID_MAP[MODEL]
SERVICE_CONFIG = SERVICE_CONFIG_MAP[MODEL]
MODEL_CONFIG = MODEL_CONFIG_MAP[MODEL]


@openai_endpoints(served_model=MODEL_ID)
@bentoml.service(**SERVICE_CONFIG)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(model=MODEL_ID, **MODEL_CONFIG)
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[
            int, Ge(128), Le(MODEL_CONFIG["max_model_len"])
        ] = MODEL_CONFIG["max_model_len"],
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
