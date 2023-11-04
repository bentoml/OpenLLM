# mypy: disable-error-code="call-arg,misc,attr-defined,type-abstract,type-arg,valid-type,arg-type"
from __future__ import annotations
import logging
import typing as t
import warnings

import _service_vars as svars
import orjson

import bentoml
import openllm

from openllm.entrypoints import hf
from openllm.entrypoints import openai

# The following warnings from bitsandbytes, and probably not that important for users to see
warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization')
warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization')
warnings.filterwarnings('ignore', message='The installed version of bitsandbytes was compiled without GPU support.')

logger = logging.getLogger(__name__)

model = svars.model
model_id = svars.model_id
adapter_map = svars.adapter_map
llm_config = openllm.AutoConfig.for_model(model)
llm = openllm.LLM[t.Any, t.Any](model_id, llm_config=llm_config, adapter_map=orjson.loads(adapter_map))
svc = bentoml.Service(name=f"llm-{llm_config['start_name']}-service", runners=[llm.runner])

_JsonInput = bentoml.io.JSON.from_sample({'prompt': '', 'llm_config': llm_config.model_dump(flatten=True)})

@svc.api(route='/v1/generate', input=_JsonInput, output=bentoml.io.JSON.from_sample(openllm.GenerationOutput.examples().unmarshal()))
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
  qa_inputs = openllm.GenerateInput.from_llm_config(llm_config)(**input_dict)
  return await llm.generate(qa_inputs.prompt, **qa_inputs.llm_config.model_dump())

@svc.api(route='/v1/generate_stream', input=_JsonInput, output=bentoml.io.Text(content_type='text/event-stream'))
async def generate_stream_v1(input_dict: dict[str, t.Any]) -> t.AsyncGenerator[str, None]:
  qa_inputs = openllm.GenerateInput.from_llm_config(llm_config)(**input_dict)
  return await llm.generate_iterator(qa_inputs.prompt, return_type='text', **qa_inputs.llm_config.model_dump())

@svc.api(route='/v1/metadata',
         input=bentoml.io.Text(),
         output=bentoml.io.JSON.from_sample({
             'model_id': llm.model_id,
             'timeout': 3600,
             'model_name': llm_config['model_name'],
             'backend': llm.runner.backend,
             'configuration': llm_config.model_dump(flatten=True),
             'prompt_template': llm.runner.prompt_template,
             'system_message': llm.runner.system_message,
         }))
def metadata_v1(_: str) -> openllm.MetadataOutput:
  return openllm.MetadataOutput(timeout=llm_config['timeout'],
                                model_name=llm_config['model_name'],
                                backend=llm_config['env']['backend_value'],
                                model_id=llm.model_id,
                                configuration=llm_config.model_dump_json().decode(),
                                prompt_template=llm.runner.prompt_template,
                                system_message=llm.runner.system_message)

# @svc.api(route='v1/completions',
#          input=bentoml.io.JSON.from_sample(openllm.utils.bentoml_cattr.unstructure(openllm.openai.CompletionRequest(prompt='What is 1+1?', model=runner.llm_type))),
#          output=bentoml.io.Text())
# async def completion_v1(input_dict: dict[str, t.Any], ctx: bentoml.Context) -> str | t.AsyncGenerator[str, None]:
#   _model = input_dict.get('model', None)
#   if _model != runner.llm_type: logger.warning("Model '%s' is not supported. Run openai.Model.list() to see all supported models.", _model)
#   prompt = input_dict.pop('prompt', None)
#   if prompt is None: raise ValueError("'prompt' should not be None.")
#   stream = input_dict.pop('stream', False)
#   config = {
#       'max_new_tokens': input_dict.pop('max_tokens', llm_config['max_new_tokens']),
#       'temperature': input_dict.pop('temperature', llm_config['temperature']),
#       'top_p': input_dict.pop('top_p', llm_config['top_p']),
#       'n': input_dict.pop('n', llm_config['n']),
#       'logprobs': input_dict.pop('logprobs', llm_config['logprobs']),
#       'echo': input_dict.pop('echo', False),
#       'stop': input_dict.pop('stop', llm_config['stop']),
#       'presence_penalty': input_dict.pop('presence_penalty', llm_config['presence_penalty']),
#       'frequency_penalty': input_dict.pop('frequency_penalty', llm_config['frequency_penalty']),
#       'best_of': input_dict.pop('best_of', llm_config['best_of']),
#   }
#
#   async def stream_response_generator(responses: t.AsyncGenerator[str, None]) -> t.AsyncGenerator[str, None]:
#     async for response in responses:
#       st = openllm.openai.CompletionResponseStream(choices=[openllm.openai.CompletionTextChoice(text=response, index=0)], model=runner.llm_type)  # TODO: logprobs, finish_reason
#       yield f'data: {orjson.dumps(openllm.utils.bentoml_cattr.unstructure(st)).decode()}\n\n'
#     yield 'data: [DONE]\n\n'
#
#   if stream:
#     ctx.response.headers['Content-Type'] = 'text/event-stream'
#     if runner.backend == 'vllm':
#       responses = runner.vllm_generate_iterator.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config)
#     else:
#       responses = runner.generate_iterator.async_stream(prompt, **config)
#     return stream_response_generator(responses)
#   else:
#     ctx.response.headers['Content-Type'] = 'application/json'
#     if runner.backend == 'vllm':
#       async for output in runner.vllm_generate.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config):
#         responses = output
#       if responses is None: raise ValueError("'responses' should not be None.")
#     else:
#       responses = await runner.generate.async_run(prompt, **config)
#
#     return orjson.dumps(
#         openllm.utils.bentoml_cattr.unstructure(
#             openllm.openai.CompletionResponse(choices=[openllm.openai.CompletionTextChoice(text=response, index=i) for i, response in enumerate(responses)],
#                                               model=runner.llm_type)  # TODO: logprobs, finish_reason and usage
#         )).decode()
#
# @svc.api(route='/v1/chat/completions',
#          input=bentoml.io.JSON.from_sample(openllm.utils.bentoml_cattr.unstructure(openllm.openai.ChatCompletionRequest(messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}], model=runner.llm_type))),
#          output=bentoml.io.Text())
# async def chat_completion_v1(input_dict: dict[str, t.Any], ctx: bentoml.Context) -> str | t.AsyncGenerator[str, None]:
#   _model = input_dict.get('model', None)
#   if _model != runner.llm_type: logger.warning("Model '%s' is not supported. Run openai.Model.list() to see all supported models.", _model)
#   prompt = openllm.openai.messages_to_prompt(input_dict['messages'], model, llm_config)
#   stream = input_dict.pop('stream', False)
#   config = {
#       'temperature': input_dict.pop('temperature', llm_config['temperature']),
#       'top_p': input_dict.pop('top_p', llm_config['top_p']),
#       'n': input_dict.pop('n', llm_config['n']),
#       'echo': input_dict.pop('echo', False),
#       'stop': input_dict.pop('stop', llm_config['stop']),
#       'max_new_tokens': input_dict.pop('max_tokens', llm_config['max_new_tokens']),
#       'presence_penalty': input_dict.pop('presence_penalty', llm_config['presence_penalty']),
#       'frequency_penalty': input_dict.pop('frequency_penalty', llm_config['frequency_penalty']),
#       '_format_chat_template': True,
#   }
#
#   async def stream_response_generator(responses: t.AsyncGenerator[str, None]) -> t.AsyncGenerator[str, None]:
#     async for response in responses:
#       st = openllm.openai.ChatCompletionResponseStream(
#           choices=[openllm.openai.ChatCompletionStreamChoice(index=0, delta=openllm.openai.Message(role='assistant', content=response), finish_reason=None)], model=runner.llm_type)
#       yield f'data: {orjson.dumps(openllm.utils.bentoml_cattr.unstructure(st)).decode()}\n\n'
#     final = openllm.openai.ChatCompletionResponseStream(
#         choices=[openllm.openai.ChatCompletionStreamChoice(index=0, delta=openllm.openai.Message(role='assistant', content=''), finish_reason='stop')], model=runner.llm_type)
#     yield f'data: {orjson.dumps(openllm.utils.bentoml_cattr.unstructure(final)).decode()}\n\n'
#     yield 'data: [DONE]\n\n'
#
#   if stream:
#     ctx.response.headers['Content-Type'] = 'text/event-stream'
#     if runner.backend == 'vllm':
#       responses = runner.vllm_generate_iterator.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config)
#     else:
#       responses = runner.generate_iterator.async_stream(prompt, **config)
#     return stream_response_generator(responses)
#   else:
#     ctx.response.headers['Content-Type'] = 'application/json'
#     if runner.backend == 'vllm':
#       async for output in runner.vllm_generate.async_stream(prompt, request_id=openllm_core.utils.gen_random_uuid(), **config):
#         responses = output
#       if responses is None: raise ValueError("'responses' should not be None.")
#     else:
#       responses = await runner.generate.async_run(prompt, **config)
#     return orjson.dumps(
#         openllm.utils.bentoml_cattr.unstructure(
#             openllm.openai.ChatCompletionResponse(
#                 choices=[openllm.openai.ChatCompletionChoice(index=i, message=openllm.openai.Message(role='assistant', content=response)) for i, response in enumerate(responses)],
#                 model=runner.llm_type)  # TODO: logprobs, finish_reason and usage
#         )).decode('utf-8')
#
# def models_v1(_: Request) -> Response:
#   return JSONResponse(openllm.utils.bentoml_cattr.unstructure(openllm.openai.ModelList(data=[openllm.openai.ModelCard(id=runner.llm_type)])), status_code=200)
#
# openai_app = Starlette(debug=True, routes=[Route('/models', models_v1, methods=['GET'])])
# svc.mount_asgi_app(openai_app, path='/v1')

openai.mount_to_svc(hf.mount_to_svc(svc, llm), llm)
