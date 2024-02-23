from __future__ import annotations
import openllm, bentoml, openllm._service_vars as svars, typing as t
from openllm_core._schemas import (
  MessageParam,
  GenerationInput,
  GenerationInputDict,
  GenerationOutput,
  MessagesConverterInput,
)
from openllm_core._typing_compat import Unpack

bentomodel = openllm.prepare_model(
  svars.model_id,
  bentomodel_tag=svars.model_tag,
  serialisation=svars.serialization,
  trust_remote_code=svars.trust_remote_code,
)
config = openllm.AutoConfig.from_bentomodel(bentomodel)
generations = GenerationInput.from_llm_config(config)


@bentoml.service(name=f"llm-{config['start_name']}-service", **svars.services_config)
class LLMService:
  bentomodel = bentomodel

  def __init__(self):
    self.llm = openllm.LLM.from_model(
      self.bentomodel,
      adapter_map=svars.adapter_map,
      max_model_len=svars.max_model_len,
      gpu_memory_utilization=svars.gpu_memory_utilization,
      trust_remote_code=svars.trust_remote_code,
    )
    llm_ref = self.llm

  @openllm.utils.api(route='/v1/generate', output=GenerationOutput, input=generations)
  async def generate_v1(self, **parameters: Unpack[GenerationInputDict]) -> GenerationOutput:
    # XXX FIX ME THIS IS BROKEN
    parameters.pop('llm_config')
    structured = generations(**parameters)
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
  async def generate_stream_v1(self, **parameters: Unpack[GenerationInputDict]) -> t.AsyncGenerator[str, None]:
    # XXX FIX ME THIS IS BROKEN
    parameters.pop('llm_config')
    structured = generations(**parameters)
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
      configuration=self.llm.config.model_dump_json(),
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


# openllm.mount_entrypoints(LLMService, llm_ref)

if __name__ == '__main__':
  LLMService.serve_http(
    reload=openllm.utils.getenv('reload', default=True),
    development_mode=openllm.utils.getenv('development', default=False),
  )
