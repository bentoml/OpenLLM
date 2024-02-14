from __future__ import annotations
import openllm, bentoml, _service_vars as svars, typing as t, types
from openllm_core._typing_compat import TypedDict
from openllm_core._schemas import MessageParam, GenerationInput, GenerationInputDict, GenerationOutput


class MessagesConverterInput(TypedDict):
  add_generation_prompt: bool
  messages: t.List[t.Dict[str, t.Any]]


# llm = openllm.LLM(
#   model_id=svars.model_id,
#   model_tag=svars.model_tag,
#   adapter_map=svars.adapter_map,
#   serialisation=svars.serialization,
#   trust_remote_code=svars.trust_remote_code,
#   max_model_len=svars.max_model_len,
#   gpu_memory_utilization=svars.gpu_memory_utilization,
#   services_config=svars.services_config,
# )

bentomodel = openllm.prepare_model(
  model_id=svars.model_id,
  bentomodel_tag=svars.model_tag,
  serialisation=svars.serialization,
  trust_remote_code=svars.trust_remote_code,
)

generations = GenerationInput.from_llm_config(openllm.AutoConfig.for_model())


def apis(llm, generations):
  @openllm.utils.api(route='/v1/generate', output=GenerationOutput, input=generations)
  async def generate_v1(self, parameters: GenerationInputDict = generations.examples) -> GenerationOutput:
    structured = generations.from_dict(**parameters)
    return await self.llm.generate(
      structured.prompt,
      structured.prompt_token_ids,  #
      structured.stop,
      structured.stop_token_ids,  #
      structured.request_id,
      structured.adapter_name,  #
      **structured.llm_config.model_dump(flatten=True),
    )

  @openllm.utils.api(route='/v1/generate_stream', input=generations)
  async def generate_stream_v1(
    self, parameters: GenerationInputDict = generations.examples
  ) -> t.AsyncGenerator[str, None]:
    structured = GenerationInput(**parameters)
    async for generated in self.llm.generate_iterator(
      structured.prompt,
      structured.prompt_token_ids,  #
      structured.stop,
      structured.stop_token_ids,  #
      structured.request_id,
      structured.adapter_name,  #
      **structured.llm_config.model_dump(flatten=True),
    ):
      yield f'data: {generated.model_dump_json()}\n\n'
    yield 'data: [DONE]\n\n'

  @openllm.utils.api(output=openllm.MetadataOutput, route='/v1/metadata')
  def metadata_v1(self) -> openllm.MetadataOutput:
    return openllm.MetadataOutput(
      timeout=self.llm.config['timeout'],
      model_name=self.llm.config['model_name'],
      backend=self.llm.__llm_backend__,
      model_id=self.llm.model_id,
      configuration=self.llm.config.model_dump_json().decode(),
    )

  @openllm.utils.api(route='/v1/helpers/messages')
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
    add_generation_prompt, messages = message['add_generation_prompt'], message['messages']
    return self.llm.tokenizer.apply_chat_template(
      messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )

  list_apis = {
    k: v for k, v in locals().items() if hasattr(v, 'func') and getattr(v.func, '__openllm_api_func__', False)
  }

  return types.new_class(
    llm.config.__class__.__name__[:-6] + 'Service',
    exec_body=lambda ns: ns.update({'llm': llm, '__module__': __name__, **list_apis}),
  )


@bentoml.service(**svars.services_config)
class LLMService:
  def __init__(self):
    self.llm = openllm.LLM.from_model(
      self.bentomodel,
      adapter_map=svars.adapter_map,
      max_model_len=svars.max_model_len,
      gpu_memory_utilization=svars.gpu_memory_utilization,
      trust_remote_code=svars.trust_remote_code,
    )

  @openllm.utils.api(route='/v1/generate', output=GenerationOutput, input=generations, media_type='application/json')
  async def generate_v1(self, parameters: GenerationInputDict = generations.examples) -> GenerationOutput:
    structured = generations.from_dict(**parameters)
    return await llm.generate(
      structured.prompt,
      structured.prompt_token_ids,  #
      structured.stop,
      structured.stop_token_ids,  #
      structured.request_id,
      structured.adapter_name,  #
      **structured.llm_config.model_dump(flatten=True),
    )

  @openllm.utils.api(route='/v1/generate_stream', input=generations)
  async def generate_stream_v1(
    self, parameters: GenerationInputDict = generations.examples
  ) -> t.AsyncGenerator[str, None]:
    structured = GenerationInput(**parameters)
    async for generated in llm.generate_iterator(
      structured.prompt,
      structured.prompt_token_ids,  #
      structured.stop,
      structured.stop_token_ids,  #
      structured.request_id,
      structured.adapter_name,  #
      **structured.llm_config.model_dump(flatten=True),
    ):
      yield f'data: {generated.model_dump_json()}\n\n'
    yield 'data: [DONE]\n\n'

  @openllm.utils.api(output=openllm.MetadataOutput, route='/v1/metadata', media_type='application/json')
  def metadata_v1(self) -> openllm.MetadataOutput:
    return openllm.MetadataOutput(
      timeout=llm.config['timeout'],
      model_name=llm.config['model_name'],
      backend=llm.__llm_backend__,
      model_id=llm.model_id,
      configuration=llm.config.model_dump_json().decode(),
    )

  @openllm.utils.api(route='/v1/helpers/messages', media_type='text/plain')
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
    add_generation_prompt, messages = message['add_generation_prompt'], message['messages']
    return llm.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)


LLMService.inner.__name__ = f"llm-{llm.config['start_name']}-service"
openllm.mount_entrypoints(LLMService, llm)

if __name__ == '__main__':
  LLMService.serve_http(
    reload=openllm.utils.getenv('reload', default=True),
    development_mode=openllm.utils.getenv('development', default=False),
  )
