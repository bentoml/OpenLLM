from __future__ import annotations

from http import HTTPStatus
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
import openllm, bentoml, logging, openllm_core as core
import _service_vars as svars, typing as t
from openllm_core._typing_compat import Annotated
from openllm_core._schemas import MessageParam, MessagesConverterInput
from openllm_core.protocol.openai import ModelCard, ModelList, ChatCompletionRequest
from _openllm_tiny._helpers import OpenAI, Error

try:
  from fastapi import FastAPI
except ImportError:
  raise ImportError("Make sure to install openllm with 'pip install openllm[openai]'") from None

logger = logging.getLogger(__name__)

bentomodel = openllm.prepare_model(
  svars.model_id,
  bentomodel_tag=svars.model_tag,
  bentomodel_version=svars.model_version,
  serialization=svars.serialisation,
  quantize=svars.quantise,
  dtype=svars.dtype,
  trust_remote_code=svars.trust_remote_code,
)
llm_config = core.AutoConfig.from_bentomodel(bentomodel)
GenerationInput = core.GenerationInput.from_config(llm_config)

app_v1 = FastAPI(debug=True, description='OpenAI Compatible API support')


@bentoml.mount_asgi_app(app_v1)
@bentoml.service(name=f"llm-{llm_config['start_name']}-service", **svars.services_config)
class LLMService:
  bentomodel = bentomodel

  def __init__(self):
    self.llm = openllm.LLM.from_model(
      self.bentomodel,
      llm_config=llm_config,
      max_model_len=svars.max_model_len,
      gpu_memory_utilization=svars.gpu_memory_utilization,
      trust_remote_code=svars.trust_remote_code,
    )
    self.openai = OpenAI(self.llm)

  @core.utils.api(route='/v1/generate')
  async def generate_v1(self, **parameters: t.Any) -> core.GenerationOutput:
    return await self.llm.generate(**GenerationInput.from_dict(parameters).model_dump())

  @core.utils.api(route='/v1/generate_stream')
  async def generate_stream_v1(self, **parameters: t.Any) -> t.AsyncGenerator[str, None]:
    async for generated in self.llm.generate_iterator(**GenerationInput.from_dict(parameters).model_dump()):
      yield f'data: {generated.model_dump_json()}\n\n'
    yield 'data: [DONE]\n\n'

  @core.utils.api(route='/v1/metadata')
  async def metadata_v1(self) -> core.MetadataOutput:
    return core.MetadataOutput(
      timeout=self.llm.config['timeout'],
      model_name=self.llm.config['model_name'],
      backend='vllm',  # deprecated
      model_id=self.llm.model_id,
      configuration=self.llm.config.model_dump_json(),
    )

  @core.utils.api(route='/v1/helpers/messages')
  def helpers_messages_v1(
    self,
    message: Annotated[t.Dict[str, t.Any], MessagesConverterInput] = MessagesConverterInput(
      add_generation_prompt=False,
      messages=[
        MessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
        MessageParam(role='user', content='Hi there!'),
        MessageParam(role='assistant', content='Yes?'),
      ],
    ),
  ) -> str:
    return self.llm._tokenizer.apply_chat_template(
      message['messages'], add_generation_prompt=message['add_generation_prompt'], tokenize=False
    )

  @app_v1.post(
    '/v1/chat/completions',
    tags=['OpenAI'],
    status_code=HTTPStatus.OK,
    summary='Given a list of messages comprising a conversation, the model will return a response.',
    operation_id='openai__chat_completions',
  )
  async def chat_completions_v1(
    self,
    raw_request: Request,
    request: ChatCompletionRequest = ChatCompletionRequest(
      messages=[
        MessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
        MessageParam(role='user', content='Hi there!'),
        MessageParam(role='assistant', content='Yes?'),
      ],
      model=core.utils.normalise_model_name(bentomodel.info.metadata['model_id']),
      n=1,
      stream=True,
    ),
  ):
    generator = await self.openai.chat_completions(request, raw_request)
    if isinstance(generator, Error):
      return JSONResponse(content=generator.model_dump(), status_code=generator.error.code)
    if request.stream is True:
      return StreamingResponse(generator, media_type='text/event-stream')
    return JSONResponse(content=generator.model_dump())

  # GET /v1/models
  @app_v1.get(
    '/v1/models',
    tags=['OpenAI'],
    status_code=HTTPStatus.OK,
    summary='Describes a model offering that can be used with the API.',
    operation_id='openai__list_models',
  )
  def list_models(self) -> ModelList:
    """
    List and describe the various models available in the API.

    You can refer to the available supported models with `openllm models` for more information.
    """
    return ModelList(
      data=[
        ModelCard(
          root=core.utils.normalise_model_name(bentomodel.info.metadata['model_id']),
          id=core.utils.normalise_model_name(bentomodel.info.metadata['model_id']),
        )
      ]
    )


LLMService.mount_asgi_app(app_v1)

if __name__ == '__main__':
  LLMService.serve_http(reload=core.utils.check_bool_env('RELOAD', False))
