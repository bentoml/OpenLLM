import uuid
import typing as t
from typing import AsyncGenerator

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated


MAX_TOKENS = 1024
PROMPT_TEMPLATE = """<s>[INST]
Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
{user_prompt} [/INST] """


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

DEFAULT_SCHEMA = """
{
  "title": "User",
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "last_name": {"type": "string"},
    "id": {"type": "integer"}
  }
}
"""

DEFAULT_USER_PROMPT = "Create a user profile with the fields name, last_name and id. name should be common English first names. last_name should be common English last names. id should be a random integer"


@bentoml.service(
    name="mistral-7b-instruct-outlines-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_TOKENS
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        prompt: str = DEFAULT_USER_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
        json_schema: t.Optional[str] = DEFAULT_SCHEMA,
        regex_string: t.Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams
        from outlines.integrations.vllm import JSONLogitsProcessor, RegexLogitsProcessor

        if json_schema is not None:
            logits_processors = [JSONLogitsProcessor(json_schema, self.engine.engine)]
        elif regex_string is not None:
            logits_processors = [RegexLogitsProcessor(regex_string, self.engine.engine)]
        else:
            logits_processors = []

        sampling_param = SamplingParams(
            max_tokens=max_tokens,
            logits_processors=logits_processors,
        )
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, sampling_param)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
