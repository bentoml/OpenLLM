from __future__ import annotations
import logging, typing as t
import bentoml, openllm, _service_vars as svars
from openllm_core._schemas import MessageParam
from bentoml.io import JSON, Text

logger = logging.getLogger(__name__)
llm = openllm.LLM[t.Any, t.Any](
  model_id=svars.model_id,
  model_tag=svars.model_tag,
  adapter_map=svars.adapter_map,  #
  serialisation=svars.serialization,
  trust_remote_code=svars.trust_remote_code,  #
  max_model_len=svars.max_model_len,
  gpu_memory_utilization=svars.gpu_memory_utilization,  #
)
svc = bentoml.Service(name=f"llm-{llm.config['start_name']}-service", runners=[llm.runner])
llm_model_class = openllm.GenerationInput.from_llm_config(llm.config)


@svc.api(route='/v1/generate', input=JSON.from_sample(llm_model_class.examples()), output=JSON.from_sample(openllm.GenerationOutput.examples()))
async def generate_v1(input_dict: dict[str, t.Any]) -> dict[str, t.Any]:
  return (await llm.generate(**llm_model_class(**input_dict).model_dump())).model_dump()


@svc.api(route='/v1/generate_stream', input=JSON.from_sample(llm_model_class.examples()), output=Text(content_type='text/event-stream'))
async def generate_stream_v1(input_dict: dict[str, t.Any]) -> t.AsyncGenerator[str, None]:
  async for it in llm.generate_iterator(**llm_model_class(**input_dict).model_dump()):
    yield f'data: {it.model_dump_json()}\n\n'
  yield 'data: [DONE]\n\n'


_Metadata = openllm.MetadataOutput(
  timeout=llm.config['timeout'],
  model_name=llm.config['model_name'],  #
  backend=llm.__llm_backend__,
  model_id=llm.model_id,
  configuration=llm.config.model_dump_json().decode(),  #
)


@svc.api(route='/v1/metadata', input=Text(), output=JSON.from_sample(_Metadata.model_dump()))
def metadata_v1(_: str) -> openllm.MetadataOutput:
  return _Metadata


class MessagesConverterInput(t.TypedDict):
  add_generation_prompt: bool
  messages: t.List[t.Dict[str, t.Any]]


@svc.api(
  route='/v1/helpers/messages',
  input=JSON.from_sample(
    MessagesConverterInput(
      add_generation_prompt=False,
      messages=[
        MessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
        MessageParam(role='user', content='Hi there!'),
        MessageParam(role='assistant', content='Yes?'),  #
      ],
    )
  ),
  output=Text(),
)
def helpers_messages_v1(message: MessagesConverterInput) -> str:
  add_generation_prompt, messages = message['add_generation_prompt'], message['messages']
  return llm.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)


openllm.mount_entrypoints(svc, llm)  # HACK: This must always be the last line in this file, as we will do some MK for OpenAPI schema.
