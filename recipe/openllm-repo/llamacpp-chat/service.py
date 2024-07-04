import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated
from llama_cpp import Llama
from typing import AsyncGenerator, Optional
from bento_constants import CONSTANT_YAML
import yaml
import fastapi
import fastapi.staticfiles
import os
from fastapi.responses import FileResponse
from typing_extensions import Annotated, Literal
import sys
import pydantic
from bentoml.io import SSE

SYS_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.If you don't know the answer to a question, please don't share false information
"""

CONSTANTS = yaml.safe_load(CONSTANT_YAML)

ENGINE_CONFIG = CONSTANTS["engine_config"]
SERVICE_CONFIG = CONSTANTS["service_config"]
OVERRIDE_CHAT_TEMPLATE = CONSTANTS.get("chat_template")

class Message(pydantic.BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

STATIC_DIR = os.path.join(os.path.dirname(__file__), "ui")

static_app = fastapi.FastAPI()
ui_app = fastapi.FastAPI()
openai_api_app = fastapi.FastAPI()

@openai_api_app.get("/models")
async def show_available_models():
    # Return the available models
    return {
        "data":[
            {
                "id": ENGINE_CONFIG["model"],
                "object": "model",
                "created": 1686935002,
                "owned_by": "bentoml",
            }
        ]
    }

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


@bentoml.mount_asgi_app(ui_app, path="/chat")
@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(**SERVICE_CONFIG)
class Phi3:
    
    def __init__(self) -> None:
        self.llm = Llama.from_pretrained(
            repo_id=ENGINE_CONFIG["model"],
            filename="*q4.gguf",
            verbose=False,
        )

    @bentoml.api(route="/api/chat")
    async def chat(
        self,
        messages: list[Message] = [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        model: str = ENGINE_CONFIG["model"],
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
        stop: Optional[list[str]] = None,
        stop_token_ids: Optional[list[int]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        light-weight chat API that takes in a list of messages and returns a response
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stop=stop,
        )

        for chunk in response:
            try:
                yield chunk["choices"][0]["delta"]["content"]
            except KeyError:
                yield ""
    
    @bentoml.api(route="/v1/chat/completions")
    async def chat_completions(
        self,
        messages: list[Message] = [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        model: str = ENGINE_CONFIG["model"],
        max_tokens: Annotated[
            int,
            Ge(128),
            Le(ENGINE_CONFIG["max_model_len"]),
        ] = ENGINE_CONFIG["max_model_len"],
        stop: Optional[list[str]] = None,
        stream: Optional[bool] = False,
        temperature: Optional[float] = 0,
        top_p: Optional[float] = 1.0,
        frequency_penalty: Optional[float] = 0.0,
    ) -> AsyncGenerator[str, None]:
        """
        Chat API that takes in a list of messages and returns a response
        """
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                stream=stream,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
            )

            for chunk in response:
                try:
                    import json
                    json_srt = json.dumps(chunk)
                    sse = SSE(data=json_srt)
                    yield sse.marshal()
                except Exception as e:
                    print(e)
                    yield SSE(data=str(e)).marshal()
            
            yield SSE(data="[DONE]").marshal()
        except Exception as e:
            yield SSE(data=str(e)).marshal()
            yield SSE(data="[DONE]").marshal()
    
if __name__ == "__main__":
    phi3 = Phi3()
    response = phi3.llm.create_chat_completion(
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": "Explain superconductors like I'm five years old"}
            ],
            max_tokens=256,
            #stream=True,
        )
    print(response)