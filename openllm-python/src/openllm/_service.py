from __future__ import annotations
import openllm, bentoml, _service_vars as svars, typing as t
from openllm_core._schemas import MessageParam, GenerationInput, GenerationInputDict, GenerationOutput, MessagesConverterInput

bentomodel = openllm.prepare_model(
  model_id=svars.model_id,
  bentomodel_tag=svars.model_tag,
  serialisation=svars.serialization,
  trust_remote_code=svars.trust_remote_code,
)
config = openllm.AutoConfig.from_bentomodel(bentomodel)
generations = GenerationInput.from_llm_config(config)

llm_ref = None

@bentoml.service(name=f"llm-{config['start_name']}-service", **svars.services_config)
class LLMService:
  def __init__(self):
    self.llm = openllm.LLM.from_model(
      self.bentomodel,
      adapter_map=svars.adapter_map,
      max_model_len=svars.max_model_len,
      gpu_memory_utilization=svars.gpu_memory_utilization,
      trust_remote_code=svars.trust_remote_code,
    )
    llm_ref = self.llm

  @openllm.utils.api(route='/v1/generate', output=GenerationOutput, input=generations, media_type='application/json')
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

  @openllm.utils.api(output=openllm.MetadataOutput, route='/v1/metadata', media_type='application/json')
  def metadata_v1(self) -> openllm.MetadataOutput:
    return openllm.MetadataOutput(
      timeout=self.llm.config['timeout'],
      model_name=self.llm.config['model_name'],
      backend=self.llm.__llm_backend__,
      model_id=self.llm.model_id,
      configuration=self.llm.config.model_dump_json().decode(),
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
    return self.llm.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)


openllm.mount_entrypoints(LLMService, llm_ref)

if __name__ == '__main__':
  LLMService.serve_http(
    reload=openllm.utils.getenv('reload', default=True),
    development_mode=openllm.utils.getenv('development', default=False),
  )
