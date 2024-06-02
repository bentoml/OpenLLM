from __future__ import annotations

import openllm, bentoml, logging, pydantic, importlib.metadata
import openllm_core as core, _service_vars as svars, typing as t

from http import HTTPStatus
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from openllm_core._typing_compat import Annotated
from openllm_core._schemas import MessageParam, MessagesConverterInput
from openllm_core.protocol.openai import ModelCard, ModelList, ChatCompletionRequest
from _openllm_tiny._helpers import OpenAI, Error

try:
  from fastapi import FastAPI
except ImportError:
  raise ImportError("Make sure to install openllm with 'pip install openllm[openai]'") from None

logger = logging.getLogger(__name__)

try:
  bentomodel = bentoml.models.get(svars.model_id.lower())
  model_id = bentomodel.path
except Exception:
  bentomodel = None
  model_id = svars.model_id
LLMConfig = core.AutoConfig.from_id(model_id, trust_remote_code=svars.trust_remote_code)
GenerationInput = core.GenerationInput.from_config(LLMConfig)
ChatMessages = [
  MessageParam(
    role='system',
    content='You are acting as Ernest Hemmingway. All of your response will follow his and his writing style ONLY.',
  ),
  MessageParam(role='user', content='Write an essay on absurdism and its impact in the 20th century.'),
]

app_v1 = FastAPI(
  debug=True,
  version=importlib.metadata.version('openllm'),
  title='OpenAI',
  description='OpenAI Compatible API support',
  contact={'name': 'BentoML Team', 'email': 'contact@bentoml.com'},
)

if core.utils.get_debug_mode():
  logger.info('service_config: %s', svars.services_config)


@bentoml.mount_asgi_app(app_v1, path='/v1')
@bentoml.service(name=f"llm-{LLMConfig['start_name']}-service", **svars.services_config)
class LLMService:
  bentomodel = bentomodel

  def __init__(self):
    self.llm = openllm.LLM.from_model(
      model_id,
      dtype=svars.dtype,
      bentomodel=bentomodel,
      revision=svars.revision,
      serialisation=svars.serialisation,
      quantise=svars.quantise,
      llm_config=LLMConfig,
      trust_remote_code=svars.trust_remote_code,
      max_model_len=svars.max_model_len,
      gpu_memory_utilization=svars.gpu_memory_utilization,
    )
    self.openai = OpenAI(self.llm)

  @core.utils.api(route='/v1/generate')
  async def generate_v1(
    self,
    prompt: str = pydantic.Field(default='What is the meaning of life?', description='Given prompt to generate from'),
    prompt_token_ids: t.Optional[t.List[int]] = None,
    llm_config: t.Dict[str, t.Any] = pydantic.Field(default=LLMConfig.model_dump(), description='LLM Config'),
  ) -> core.GenerationOutput:
    return await self.llm.generate(prompt=prompt, prompt_token_ids=prompt_token_ids, **llm_config)

  @core.utils.api(route='/v1/generate_stream')
  async def generate_stream_v1(
    self,
    prompt: str = pydantic.Field(default='What is the meaning of life?', description='Given prompt to generate from'),
    prompt_token_ids: t.Optional[t.List[int]] = None,
    llm_config: t.Dict[str, t.Any] = pydantic.Field(default=LLMConfig.model_dump(), description='LLM Config'),
  ) -> t.AsyncGenerator[str, None]:
    _config = LLMConfig.model_construct_env(**core.utils.dict_filter_none(llm_config))
    previous_texts = [''] * _config['n']
    previous_num_tokens = [0] * _config['n']
    finish_reason_sent = [False] * _config['n']

    async for generations in self.llm.generate_iterator(
      prompt=prompt, prompt_token_ids=prompt_token_ids, **_config.model_dump()
    ):
      for output in generations.outputs:
        i = output.index
        if finish_reason_sent[i]:
          continue
        delta_token_ids = output.token_ids[previous_num_tokens[i] :]
        delta_text = output.text[len(previous_texts[i]) :]
        previous_texts[i] = output.text
        previous_num_tokens[i] = len(output.token_ids)
        output.text = delta_text
        output.token_ids = delta_token_ids
        if output.finish_reason is not None:
          finish_reason_sent[i] = True
      yield f'data: {core.GenerationOutput.from_vllm(generations).model_dump_json()}\n\n'
    yield 'data: [DONE]\n\n'

  @core.utils.api(route='/v1/metadata')
  async def metadata_v1(self) -> core.MetadataOutput:
    return core.MetadataOutput(
      timeout=self.llm.config['timeout'],
      model_name=self.llm.config['model_name'],
      trust_remote_code=svars.trust_remote_code,
      model_id=self.llm.model_id,
      configuration=self.llm.config,
      serialisation=svars.serialisation,
      quantise=svars.quantise,
    )

  @core.utils.api(route='/v1/helpers/messages')
  def helpers_messages_v1(
    self,
    message: Annotated[
      t.Dict[str, t.Any],
      pydantic.Field(default=MessagesConverterInput(add_generation_prompt=True, messages=ChatMessages)),
    ],
    tokenize: bool = pydantic.Field(
      default=False, description='Whether to return tokenized version of resulted prompt.'
    ),
  ) -> str:
    return self.llm._tokenizer.apply_chat_template(
      message['messages'], add_generation_prompt=message['add_generation_prompt'], tokenize=tokenize
    )

  @app_v1.post(
    '/chat/completions',
    tags=['OpenAI'],
    status_code=HTTPStatus.OK,
    summary='Given a list of messages comprising a conversation, the model will return a response.',
    operation_id='openai__chat_completions',
  )
  async def chat_completions_v1(
    self,
    raw_request: Request,
    request: ChatCompletionRequest = ChatCompletionRequest(
      messages=ChatMessages, model=core.utils.normalise_model_name(model_id), n=1, stream=True
    ),
  ):
    generator = await self.openai.chat_completions(request, raw_request)
    if isinstance(generator, Error):
      # NOTE: safe to cast here as we know it's an error
      return JSONResponse(content=generator.model_dump(), status_code=int(t.cast(str, generator.error.code)))
    if request.stream is True:
      return StreamingResponse(generator, media_type='text/event-stream')
    return JSONResponse(content=generator.model_dump())

  # GET /v1/models
  @app_v1.get(
    '/models',
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
      data=[ModelCard(root=core.utils.normalise_model_name(model_id), id=core.utils.normalise_model_name(model_id))]
    )
