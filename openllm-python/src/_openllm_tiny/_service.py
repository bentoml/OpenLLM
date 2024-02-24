from __future__ import annotations
from http import HTTPStatus
import openllm, bentoml, logging, openllm_core as core
import _service_vars as svars, typing as t
from openllm_core._typing_compat import Unpack
from openllm_core._schemas import MessageParam, MessagesConverterInput
from openllm_core.exceptions import ModelNotFound
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from openllm_core.protocol.openai import ChatCompletionRequest, ChatCompletionResponse, LogProbs, ModelCard, ModelList
from starlette.routing import Route

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
Generation = core.GenerationInput.from_config(llm_config)


def create_logprobs(
  token_ids: list[int],
  top_logprobs: list[dict[int, float] | None] | None = None,  #
  num_output_top_logprobs: int | None = None,
  initial_text_offset: int = 0,
  *,
  llm: openllm.LLM,
) -> LogProbs:
  # Create OpenAI-style logprobs.
  logprobs = LogProbs()
  last_token_len = 0
  if num_output_top_logprobs:
    logprobs.top_logprobs = []
  for i, token_id in enumerate(token_ids):
    step_top_logprobs = top_logprobs[i]
    token_logprob = None
    if step_top_logprobs is not None:
      token_logprob = step_top_logprobs[token_id]
    token = llm.tokenizer.convert_ids_to_tokens(token_id)
    logprobs.tokens.append(token)
    logprobs.token_logprobs.append(token_logprob)
    if len(logprobs.text_offset) == 0:
      logprobs.text_offset.append(initial_text_offset)
    else:
      logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
    last_token_len = len(token)
    if num_output_top_logprobs:
      logprobs.top_logprobs.append(
        {llm.tokenizer.convert_ids_to_tokens(i): p for i, p in step_top_logprobs.items()}
        if step_top_logprobs
        else None
      )
  return logprobs


@bentoml.service(name=f"llm-{llm_config['start_name']}-service", **svars.services_config)
class Service:
  bentomodel = bentomodel

  def __init__(self):
    self.llm = openllm.LLM.from_model(
      self.bentomodel,
      llm_config=llm_config,
      max_model_len=svars.max_model_len,
      gpu_memory_utilization=svars.gpu_memory_utilization,
      trust_remote_code=svars.trust_remote_code,
    )

  @core.utils.api(route='/v1/generate', input=Generation, output=core.GenerationOutput)
  async def generate_v1(self, **parameters: Unpack[core.GenerationInputDict]) -> core.GenerationOutput:
    return await self.llm.generate(**Generation.from_dict(parameters).model_dump())

  @core.utils.api(route='/v1/generate_stream', input=Generation)
  async def generate_stream_v1(self, **parameters: Unpack[core.GenerationInputDict]) -> t.AsyncGenerator[str, None]:
    async for generated in self.llm.generate_iterator(**Generation.from_dict(parameters).model_dump()):
      yield f'data: {generated.model_dump_json()}\n\n'
    yield 'data: [DONE]\n\n'

  @core.utils.api(route='/v1/metadata', output=core.MetadataOutput)
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
    message: MessagesConverterInput = MessagesConverterInput(
      add_generation_prompt=False,
      messages=[
        MessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
        MessageParam(role='user', content='Hi there!'),
        MessageParam(role='assistant', content='Yes?'),
      ],
    ),
  ) -> str:
    return self.llm.tokenizer.apply_chat_template(
      message['messages'], add_generation_prompt=message['add_generation_prompt'], tokenize=False
    )

  # OpenAI Compatible endpoints
  @core.utils.api(route='/v1/chat/completions', input=ChatCompletionRequest)
  async def chat_completions_v1(
    self, request: ChatCompletionRequest
  ) -> t.AsyncGenerator[ChatCompletionResponse, None]:
    logger.debug('Received chat completion request: %s', request)
    if request.model != self.llm.model_id:
      raise ModelNotFound(
        f"Model '{request.model}' does not exists. Try 'GET /v1/models' to see available models.\nTip: If you are migrating from OpenAI, make sure to update your 'model' parameters in the request."
      )


# GET /v1/models
def list_models(_: Request) -> JSONResponse:
  return JSONResponse(
    ModelList(data=[ModelCard(id=core.utils.normalise_model_name(t.cast(str, bentomodel.info.metadata['model_id'])))]),
    status_code=HTTPStatus.OK.value,
  )


app = Starlette(debug=True, routes=[Route('/models', list_models, methods=['GET'])])
Service.mount_asgi_app(app, '/v1')

if __name__ == '__main__':
  Service.serve_http()
