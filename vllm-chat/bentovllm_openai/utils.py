from __future__ import annotations

import asyncio
import typing as t

from _bentoml_sdk.service.factory import Service
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .protocol import ChatCompletionRequest, CompletionRequest, ErrorResponse

T = t.TypeVar("T", bound=object)

if t.TYPE_CHECKING:
    from vllm import AsyncLLMEngine


def openai_endpoints(
    served_model: str,
    response_role: str = "assistant",
    chat_template: t.Optional[str] = None,
    chat_template_model_id: t.Optional[str] = None,
):

    def openai_wrapper(svc: Service[T]):

        cls = svc.inner
        app = FastAPI()

        class new_cls(cls):

            def __init__(self):

                super().__init__()

                # we need to import bentoml before vllm so
                # `prometheus_client` won't cause import troubles
                # That's also why we put these codes inside class's
                # `__init__` function
                import bentoml
                from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
                from vllm.entrypoints.openai.serving_completion import (
                    OpenAIServingCompletion,
                )

                # https://github.com/vllm-project/vllm/issues/2683
                class PatchedOpenAIServingChat(OpenAIServingChat):
                    def __init__(
                        self,
                        engine: AsyncLLMEngine,
                        served_model: str,
                        response_role: str,
                        chat_template=None,
                    ):
                        super(OpenAIServingChat, self).__init__(
                            engine=engine,
                            served_model=served_model,
                            lora_modules=None,
                        )
                        self.response_role = response_role
                        try:
                            event_loop = asyncio.get_running_loop()
                        except RuntimeError:
                            event_loop = None

                        if event_loop is not None and event_loop.is_running():
                            event_loop.create_task(
                                self._load_chat_template(chat_template)
                            )
                        else:
                            asyncio.run(self._load_chat_template(chat_template))

                    async def _load_chat_template(self, chat_template):
                        # Simply making this function async is usually already enough to give the parent
                        # class time to load the tokenizer (so usually no sleeping happens here)
                        # However, it feels safer to be explicit about this since asyncio does not
                        # guarantee the order in which scheduled tasks are run
                        while self.tokenizer is None:
                            await asyncio.sleep(0.1)
                        return super()._load_chat_template(chat_template)

                self.openai_serving_completion = OpenAIServingCompletion(
                    engine=self.engine,
                    served_model=served_model,
                )

                self.chat_template = chat_template
                if self.chat_template is None and chat_template_model_id is not None:
                    from transformers import AutoTokenizer

                    _tokenizer = AutoTokenizer.from_pretrained(chat_template_model_id)
                    self.chat_template = _tokenizer.chat_template

                self.openai_serving_chat = PatchedOpenAIServingChat(
                    engine=self.engine,
                    served_model=served_model,
                    response_role=response_role,
                    chat_template=self.chat_template,
                )

                @app.get("/models")
                async def show_available_models():
                    models = await self.openai_serving_chat.show_available_models()
                    return JSONResponse(content=models.model_dump())

                @app.post("/chat/completions")
                async def create_chat_completion(
                    request: ChatCompletionRequest, raw_request: Request
                ):
                    generator = await self.openai_serving_chat.create_chat_completion(
                        request, raw_request
                    )
                    if isinstance(generator, ErrorResponse):
                        return JSONResponse(
                            content=generator.model_dump(), status_code=generator.code
                        )
                    if request.stream:
                        return StreamingResponse(
                            content=generator, media_type="text/event-stream"
                        )
                    else:
                        return JSONResponse(content=generator.model_dump())

                @app.post("/completions")
                async def create_completion(
                    request: CompletionRequest, raw_request: Request
                ):
                    generator = await self.openai_serving_completion.create_completion(
                        request, raw_request
                    )
                    if isinstance(generator, ErrorResponse):
                        return JSONResponse(
                            content=generator.model_dump(), status_code=generator.code
                        )
                    if request.stream:
                        return StreamingResponse(
                            content=generator, media_type="text/event-stream"
                        )
                    else:
                        return JSONResponse(content=generator.model_dump())

        new_cls.__name__ = "%s_OpenAI" % cls.__name__
        svc.inner = new_cls
        svc.mount_asgi_app(app, path="/v1/")
        return svc

    return openai_wrapper


# helper function to make a httpx client for BentoML service
def _make_httpx_client(url, svc):

    from urllib.parse import urlparse

    import httpx
    from bentoml._internal.utils.uri import uri_to_path

    timeout = svc.config["traffic"]["timeout"]
    headers = {"Runner-Name": svc.name}
    parsed = urlparse(url)
    transport = None
    target_url = url

    if parsed.scheme == "file":
        uds = uri_to_path(url)
        transport = httpx.HTTPTransport(uds=uds)
        target_url = "http://127.0.0.1:3000"
    elif parsed.scheme == "tcp":
        target_url = f"http://{parsed.netloc}"

    return (
        httpx.Client(
            transport=transport,
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
        ),
        target_url,
    )
