import bentoml
import openllm
import openllm_core
import os
import typing as t

#run `openllm models` to find more model IDs of llama2
MODEL_ID = "NousResearch/llama-2-7b-chat-hf"  #@param ["NousResearch/llama-2-7b-chat-hf", "NousResearch/llama-2-13b-chat-hf","NousResearch/llama-2-70b-chat-hf"]
BACKEND = "pt"  #@param ["pt", "vllm"]


os.environ['OPENLLM_MODEL_ID'] = MODEL_ID
os.environ['OPENLLM_BACKEND'] = BACKEND

model = "llama"

llm_config = openllm.AutoConfig.for_model(model)
llm_runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(name="llama-service", runners=[llm_runner])

_JsonInput = bentoml.io.JSON.from_sample({'prompt': '', 'llm_config': llm_config.model_dump(flatten=True), 'adapter_name': None})


@svc.on_startup
def download(_: bentoml.Context):
  llm_runner.download_model()

@svc.api(route='/v1/generate', input=_JsonInput, output=bentoml.io.JSON.from_sample({'responses': [], 'configuration': llm_config.model_dump(flatten=True)}))
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
  qa_inputs = openllm.GenerationInput.from_llm_config(llm_config)(**input_dict)
  config = qa_inputs.llm_config.model_dump()
  if llm_runner.backend == 'vllm':
    responses = await llm_runner.vllm_generate.async_run(qa_inputs.prompt, adapter_name=qa_inputs.adapter_name, request_id=openllm_core.utils.gen_random_uuid(), **config)
  else:
    responses = await llm_runner.generate.async_run(qa_inputs.prompt, adapter_name=qa_inputs.adapter_name, **config)
  return openllm.GenerationOutput(responses=responses, configuration=config)

@svc.api(route='/v1/generate_stream', input=_JsonInput, output=bentoml.io.Text(content_type='text/event-stream'))
async def generate_stream_v1(input_dict: dict[str, t.Any]) -> t.AsyncGenerator[str, None]:
  echo = input_dict.pop('echo', False)
  qa_inputs = openllm.GenerationInput.from_llm_config(llm_config)(**input_dict)
  if llm_runner.backend == 'vllm':
    return llm_runner.vllm_generate_iterator.async_stream(qa_inputs.prompt,
                                                      adapter_name=qa_inputs.adapter_name,
                                                      echo=echo,
                                                      request_id=openllm_core.utils.gen_random_uuid(),
                                                      **qa_inputs.llm_config.model_dump())
  else:
    return llm_runner.generate_iterator.async_stream(qa_inputs.prompt, adapter_name=qa_inputs.adapter_name, echo=echo, **qa_inputs.llm_config.model_dump())