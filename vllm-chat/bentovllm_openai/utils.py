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
        model_id: str,
        response_role: str = "assistant",
        served_model_names: t.Optional[list[str]] = None,
        chat_template: t.Optional[str] = None,
        chat_template_model_id: t.Optional[str] = None,
        default_completion_parameters: t.Optional[t.Dict[str, t.Any]] = None,
        default_chat_completion_parameters: t.Optional[t.Dict[str, t.Any]] = None,
):

    if served_model_names is None:
        served_model_names = [model_id]

    def openai_wrapper(svc: Service[T]):

        cls = svc.inner
        app = FastAPI()

        # make sure default_*_parameters are in valid format
        if default_completion_parameters is not None:
            assert "prompt" not in default_completion_parameters
            assert CompletionRequest(
                prompt="", model="", **default_completion_parameters
            )

        if default_chat_completion_parameters is not None:
            assert "messages" not in default_chat_completion_parameters
            assert ChatCompletionRequest(
                messages=[], model="", **default_chat_completion_parameters
            )

        class new_cls(cls):

            def __init__(self):

                super().__init__()

                # we need to import bentoml before vllm so
                # `prometheus_client` won't cause import troubles
                # That's also why we put these codes inside class's
                # `__init__` function
                import bentoml

                from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
                from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

                # we can do this because worker/engine_user_ray is always False for us
                model_config = self.engine.engine.get_model_config()

                self.openai_serving_completion = OpenAIServingCompletion(
                    engine=self.engine,
                    served_model_names=served_model_names,
                    model_config=model_config,
                    lora_modules=None,
                )

                self.chat_template = chat_template
                if self.chat_template is None and chat_template_model_id is not None:
                    from transformers import AutoTokenizer
                    _tokenizer = AutoTokenizer.from_pretrained(chat_template_model_id)
                    self.chat_template = _tokenizer.chat_template

                self.openai_serving_chat = OpenAIServingChat(
                    engine=self.engine,
                    served_model_names=served_model_names,
                    response_role=response_role,
                    chat_template=self.chat_template,
                    model_config=model_config,
                )

                @app.get("/models")
                async def show_available_models():
                    models = await self.openai_serving_chat.show_available_models()
                    return JSONResponse(content=models.model_dump())

                @app.post("/chat/completions")
                async def create_chat_completion(
                        request: ChatCompletionRequest,
                        raw_request: Request
                ):
                    if default_chat_completion_parameters is not None:
                        for k, v in default_chat_completion_parameters.items():
                            if k not in request.__fields_set__:
                                setattr(request, k, v)
                    generator = await self.openai_serving_chat.create_chat_completion(
                        request, raw_request)
                    if isinstance(generator, ErrorResponse):
                        return JSONResponse(content=generator.model_dump(),
                                            status_code=generator.code)
                    if request.stream:
                        return StreamingResponse(content=generator,
                                                 media_type="text/event-stream")
                    else:
                        return JSONResponse(content=generator.model_dump())

                @app.post("/completions")
                async def create_completion(request: CompletionRequest, raw_request: Request):
                    if default_completion_parameters is not None:
                        for k, v in default_completion_parameters.items():
                            if k not in request.__fields_set__:
                                setattr(request, k, v)
                    generator = await self.openai_serving_completion.create_completion(
                        request, raw_request)
                    if isinstance(generator, ErrorResponse):
                        return JSONResponse(content=generator.model_dump(),
                                            status_code=generator.code)
                    if request.stream:
                        return StreamingResponse(content=generator,
                                                 media_type="text/event-stream")
                    else:
                        return JSONResponse(content=generator.model_dump())

        new_cls.__name__ = "%s_OpenAI" % cls.__name__
        svc.inner = new_cls
        svc.mount_asgi_app(app, path="/v1/")
        return svc

    return openai_wrapper


# helper function to make a httpx client for BentoML service
def _make_httpx_client(url, svc):

    import httpx
    from urllib.parse import urlparse
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

    return httpx.Client(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
        headers=headers,
    ), target_url
