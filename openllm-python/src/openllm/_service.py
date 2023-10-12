# mypy: disable-error-code="call-arg,misc,attr-defined,type-abstract,type-arg,valid-type,arg-type"
from __future__ import annotations
import typing as t
import warnings

import _service_vars as svars
import orjson

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

import bentoml
import openllm
import openllm_core

if t.TYPE_CHECKING:
  from starlette.requests import Request
  from starlette.responses import Response

  from bentoml._internal.runner.runner import AbstractRunner
  from bentoml._internal.runner.runner import RunnerMethod
  from openllm_core._typing_compat import TypeAlias
  _EmbeddingMethod: TypeAlias = RunnerMethod[t.Union[bentoml.Runnable, openllm.LLMRunnable[t.Any, t.Any]], [t.List[str]], t.Sequence[openllm.EmbeddingsOutput]]

# The following warnings from bitsandbytes, and probably not that important for users to see
warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization')
warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization')
warnings.filterwarnings('ignore', message='The installed version of bitsandbytes was compiled without GPU support.')

model = svars.model
model_id = svars.model_id
adapter_map = svars.adapter_map
llm_config = openllm.AutoConfig.for_model(model)
runner = openllm.Runner(model, llm_config=llm_config, model_id=model_id, ensure_available=False, adapter_map=orjson.loads(adapter_map))
generic_embedding_runner = bentoml.Runner(openllm.GenericEmbeddingRunnable,  # XXX: remove arg-type once bentoml.Runner is correct set with type
                                          name='llm-generic-embedding',
                                          scheduling_strategy=openllm_core.CascadingResourceStrategy,
                                          max_batch_size=32,
                                          max_latency_ms=300)
runners: list[AbstractRunner] = [runner]
if not runner.supports_embeddings: runners.append(generic_embedding_runner)
svc = bentoml.Service(name=f"llm-{llm_config['start_name']}-service", runners=runners)

_JsonInput = bentoml.io.JSON.from_sample({'prompt': '', 'llm_config': llm_config.model_dump(flatten=True), 'adapter_name': None})

@svc.api(route='/v1/generate', input=_JsonInput, output=bentoml.io.JSON.from_sample({'responses': [], 'configuration': llm_config.model_dump(flatten=True)}))
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
  echo = input_dict.pop('echo', False)
  qa_inputs = openllm.GenerationInput.from_llm_config(llm_config)(**input_dict)
  config = qa_inputs.llm_config.model_dump()
  if runner.backend == 'vllm':
    async for output in runner.vllm_generate.async_stream(qa_inputs.prompt, adapter_name=qa_inputs.adapter_name, echo=echo, request_id=openllm_core.utils.gen_random_uuid(), **config):
      responses = output
    if responses is None: raise ValueError("'responses' should not be None.")
  else:
    responses = await runner.generate.async_run(qa_inputs.prompt, adapter_name=qa_inputs.adapter_name, **config)
  return openllm.GenerationOutput(responses=responses, configuration=config)

@svc.api(route='/v1/generate_stream', input=_JsonInput, output=bentoml.io.Text(content_type='text/event-stream'))
async def generate_stream_v1(input_dict: dict[str, t.Any]) -> t.AsyncGenerator[str, None]:
  echo = input_dict.pop('echo', False)
  qa_inputs = openllm.GenerationInput.from_llm_config(llm_config)(**input_dict)
  if runner.backend == 'vllm':
    return runner.vllm_generate_iterator.async_stream(qa_inputs.prompt,
                                                      adapter_name=qa_inputs.adapter_name,
                                                      echo=echo,
                                                      request_id=openllm_core.utils.gen_random_uuid(),
                                                      **qa_inputs.llm_config.model_dump())
  else:
    return runner.generate_iterator.async_stream(qa_inputs.prompt, adapter_name=qa_inputs.adapter_name, echo=echo, **qa_inputs.llm_config.model_dump())

@svc.api(route='v1/completions',
         input=bentoml.io.JSON.from_sample(openllm.utils.bentoml_cattr.unstructure(openllm.openai.CompletionRequest(prompt='What is 1+1?', model=runner.llm_type))),
         output=bentoml.io.Text())
async def completion_v1(input_dict: dict[str, t.Any], ctx: bentoml.Context) -> str | t.AsyncGenerator[str, None]:
  prompt = input_dict.pop('prompt', None)
  if prompt is None: raise ValueError("'prompt' should not be None.")
  stream = input_dict.pop('stream', False)
  config = {
      'max_new_tokens': input_dict.pop('max_tokens', llm_config['max_new_tokens']),
      'temperature': input_dict.pop('temperature', llm_config['temperature']),
      'top_p': input_dict.pop('top_p', llm_config['top_p']),
      'n': input_dict.pop('n', llm_config['n']),
      'logprobs': input_dict.pop('logprobs', llm_config['logprobs']),
      'echo': input_dict.pop('echo', False),
      'stop': input_dict.pop('stop', llm_config['stop']),
      'presence_penalty': input_dict.pop('presence_penalty', llm_config['presence_penalty']),
      'frequency_penalty': input_dict.pop('frequency_penalty', llm_config['frequency_penalty']),
      'best_of': input_dict.pop('best_of', llm_config['best_of']),
  }

  async def stream_response_generator(responses: t.AsyncGenerator[str, None]) -> t.AsyncGenerator[str, None]:
    async for response in responses:
      st = openllm.openai.CompletionResponseStream(choices=[openllm.openai.CompletionTextChoice(text=response, index=0)], model=runner.llm_type)  # TODO: logprobs, finish_reason
      yield f'data: {orjson.dumps(openllm.utils.bentoml_cattr.unstructure(st)).decode()}\n\n'
    yield 'data: [DONE]\n\n'

  if stream:
    ctx.response.headers['Content-Type'] = 'text/event-stream'
    if runner.backend == 'vllm':
      responses = runner.vllm_generate_iterator.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config)
    else:
      responses = runner.generate_iterator.async_stream(prompt, **config)
    return stream_response_generator(responses)
  else:
    ctx.response.headers['Content-Type'] = 'application/json'
    if runner.backend == 'vllm':
      async for output in runner.vllm_generate.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config):
        responses = output
      if responses is None: raise ValueError("'responses' should not be None.")
    else:
      responses = await runner.generate.async_run(prompt, **config)

    return orjson.dumps(
        openllm.utils.bentoml_cattr.unstructure(
            openllm.openai.CompletionResponse(choices=[openllm.openai.CompletionTextChoice(text=response, index=i) for i, response in enumerate(responses)],
                                              model=runner.llm_type)  # TODO: logprobs, finish_reason and usage
        )).decode()

@svc.api(route='/v1/chat/completions',
         input=bentoml.io.JSON.from_sample(
             openllm.utils.bentoml_cattr.unstructure(
                 openllm.openai.ChatCompletionRequest(messages=[{
                     'role': 'system',
                     'content': 'You are a helpful assistant.'
                 }, {
                     'role': 'user',
                     'content': 'Hello!'
                 }], model=runner.llm_type))),
         output=bentoml.io.Text())
async def chat_completion_v1(input_dict: dict[str, t.Any], ctx: bentoml.Context) -> str | t.AsyncGenerator[str, None]:
  prompt = openllm.openai.messages_to_prompt(input_dict['messages'])
  stream = input_dict.pop('stream', False)
  config = {
      'temperature': input_dict.pop('temperature', llm_config['temperature']),
      'top_p': input_dict.pop('top_p', llm_config['top_p']),
      'n': input_dict.pop('n', llm_config['n']),
      'echo': input_dict.pop('echo', False),
      'stop': input_dict.pop('stop', llm_config['stop']),
      'max_new_tokens': input_dict.pop('max_tokens', llm_config['max_new_tokens']),
      'presence_penalty': input_dict.pop('presence_penalty', llm_config['presence_penalty']),
      'frequency_penalty': input_dict.pop('frequency_penalty', llm_config['frequency_penalty']),
  }

  async def stream_response_generator(responses: t.AsyncGenerator[str, None]) -> t.AsyncGenerator[str, None]:
    async for response in responses:
      st = openllm.openai.ChatCompletionResponseStream(
          choices=[openllm.openai.ChatCompletionStreamChoice(index=0, delta=openllm.openai.Message(role='assistant', content=response), finish_reason=None)], model=runner.llm_type)
      yield f'data: {orjson.dumps(openllm.utils.bentoml_cattr.unstructure(st)).decode()}\n\n'
    final = openllm.openai.ChatCompletionResponseStream(
        choices=[openllm.openai.ChatCompletionStreamChoice(index=0, delta=openllm.openai.Message(role='assistant', content=''), finish_reason='stop')], model=runner.llm_type)
    yield f'data: {orjson.dumps(openllm.utils.bentoml_cattr.unstructure(final)).decode()}\n\n'
    yield 'data: [DONE]\n\n'

  if stream:
    ctx.response.headers['Content-Type'] = 'text/event-stream'
    if runner.backend == 'vllm':
      responses = runner.vllm_generate_iterator.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config)
    else:
      responses = runner.generate_iterator.async_stream(prompt, **config)
    return stream_response_generator(responses)
  else:
    ctx.response.headers['Content-Type'] = 'application/json'
    if runner.backend == 'vllm':
      async for output in runner.vllm_generate.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config):
        responses = output
      if responses is None: raise ValueError("'responses' should not be None.")
    else:
      responses = await runner.generate.async_run(prompt, **config)
    return orjson.dumps(
        openllm.utils.bentoml_cattr.unstructure(
            openllm.openai.ChatCompletionResponse(choices=[
                openllm.openai.ChatCompletionChoice(index=i, message=openllm.openai.Message(role='assistant', content=response)) for i, response in enumerate(responses)
            ],
                                                  model=model)  # TODO: logprobs, finish_reason and usage
        )).decode('utf-8')

@svc.api(route='/v1/metadata',
         input=bentoml.io.Text(),
         output=bentoml.io.JSON.from_sample({
             'model_id': runner.llm.model_id,
             'timeout': 3600,
             'model_name': llm_config['model_name'],
             'backend': runner.backend,
             'configuration': llm_config.model_dump(flatten=True),
             'supports_embeddings': runner.supports_embeddings,
             'supports_hf_agent': runner.supports_hf_agent,
             'prompt_template': runner.prompt_template,
             'system_message': runner.system_message,
         }))
def metadata_v1(_: str) -> openllm.MetadataOutput:
  return openllm.MetadataOutput(timeout=llm_config['timeout'],
                                model_name=llm_config['model_name'],
                                backend=llm_config['env']['backend_value'],
                                model_id=runner.llm.model_id,
                                configuration=llm_config.model_dump_json().decode(),
                                supports_embeddings=runner.supports_embeddings,
                                supports_hf_agent=runner.supports_hf_agent,
                                prompt_template=runner.prompt_template,
                                system_message=runner.system_message,
                                )

@svc.api(route='/v1/embeddings',
         input=bentoml.io.JSON.from_sample(['Hey Jude, welcome to the jungle!', 'What is the meaning of life?']),
         output=bentoml.io.JSON.from_sample({
             'embeddings': [
                 0.007917795330286026, -0.014421648345887661, 0.00481307040899992, 0.007331526838243008, -0.0066398633643984795, 0.00945580005645752, 0.0087016262114048, -0.010709521360695362,
                 0.012635177001357079, 0.010541186667978764, -0.00730888033285737, -0.001783102168701589, 0.02339819073677063, -0.010825827717781067, -0.015888236463069916,
                 0.01876218430697918, 0.0076906150206923485, 0.0009032754460349679, -0.010024012066423893, 0.01090280432254076, -0.008668390102684498, 0.02070549875497818,
                 0.0014594447566196322, -0.018775740638375282, -0.014814382418990135, 0.01796768605709076
             ],
             'num_tokens': 20
         }))
async def embeddings_v1(phrases: list[str]) -> list[openllm.EmbeddingsOutput]:
  embed_call: _EmbeddingMethod = runner.embeddings if runner.supports_embeddings else generic_embedding_runner.encode  # type: ignore[type-arg,assignment,valid-type]
  return await embed_call.async_run(phrases)

if runner.supports_hf_agent:

  async def hf_agent(request: Request) -> Response:
    json_str = await request.body()
    try:
      input_data = openllm.utils.bentoml_cattr.structure(orjson.loads(json_str), openllm.HfAgentInput)
    except orjson.JSONDecodeError as err:
      raise openllm.exceptions.OpenLLMException(f'Invalid JSON input received: {err}') from None
    stop = input_data.parameters.pop('stop', ['\n'])
    try:
      return JSONResponse(await runner.generate_one.async_run(input_data.inputs, stop, **input_data.parameters), status_code=200)
    except NotImplementedError:
      return JSONResponse(f"'{model}' is currently not supported with HuggingFace agents.", status_code=500)

  hf_app = Starlette(debug=True, routes=[Route('/agent', hf_agent, methods=['POST'])])
  svc.mount_asgi_app(hf_app, path='/hf')

# general metadata app
async def list_adapter_v1(_: Request) -> Response:
  res: dict[str, t.Any] = {}
  if runner.peft_adapters['success'] is True:
    res['result'] = {k: v.to_dict() for k, v in runner.peft_adapters['result'].items()}
  res.update({'success': runner.peft_adapters['success'], 'error_msg': runner.peft_adapters['error_msg']})
  return JSONResponse(res, status_code=200)

adapters_app_v1 = Starlette(debug=True, routes=[Route('/adapters', list_adapter_v1, methods=['GET'])])
svc.mount_asgi_app(adapters_app_v1, path='/v1')
