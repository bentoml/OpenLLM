from __future__ import annotations
import types, sys, typing as t
import openllm, bentoml
from openllm_core._schemas import MessageParam, GenerationInput, GenerationInputDict, GenerationOutput
from openllm_core.exceptions import MissingDependencyError
from openllm_core._typing_compat import Unpack
from openllm_core.utils import gen_random_uuid

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import M, T
  from _bentoml_sdk.service.config import ServiceConfig
  from _bentoml_sdk.service.dependency import Dependency
  from ._runners import Runner

try:
  from _bentoml_sdk import Service
  from _bentoml_sdk.service.config import validate
  from _bentoml_sdk.service.dependency import depends
except ImportError:
  raise MissingDependencyError('Requires bentoml>=1.2 to be installed. Do "pip install -U "bentoml>=1.2""') from None

class LLMServiceProtocol(t.Protocol[M, T]):
  llm: openllm.LLM[M, T]
  runner: Dependency[Runner[M, T]]
  @bentoml.api
  async def generate_v1(self, input_dict: GenerationInputDict = ...) -> GenerationOutput: ...
  @bentoml.api
  async def generate_stream_v1(self, input_dict: GenerationInputDict = ...) -> t.AsyncGenerator[str, None]: ...
  @bentoml.api
  def metadata_v1(self) -> openllm.MetadataOutput: ...
  @bentoml.api
  def helpers_messages_v1(self, message: MessagesConverterInput = ...) -> str: ...

def service(llm: openllm.LLM[M, T], /, **attrs: Unpack[ServiceConfig]) -> Service[LLMServiceProtocol[M, T]]:
  generation_input_cls = GenerationInput.from_llm_config(llm.config)

  @openllm.utils.api(route='/v1/generate', output=GenerationOutput, input=generation_input_cls)
  async def generate_v1(self: LLMService, parameters: GenerationInputDict = generation_input_cls.examples) -> GenerationOutput:
    structured = GenerationInput(**parameters)
    config = self.llm.config.model_construct_env(**structured.llm_config)
    texts, token_ids = [[]] * config['n'], [[]] * config['n']
    async for generated in self.generate_stream_v1(parameters):
      if generated == 'data: [DONE]\n\n': break
      result = GenerationOutput.from_runner(generated)
      for output in result.outputs:
        texts[output.index].append(output.text)
        token_ids[output.index].extend(output.token_ids)
    if result is None: raise RuntimeError('No result is returned.')
    return result.with_options(
      prompt=structured.prompt,
      outputs=[output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index]) for output in result.outputs]
    )

  @openllm.utils.api(route='/v1/generate_stream', input=generation_input_cls)
  async def generate_stream_v1(self: LLMService, parameters: GenerationInputDict = generation_input_cls.examples) -> t.AsyncGenerator[str, None]:
    structured = GenerationInput(**parameters)
    if structured.adapter_name is not None and self.llm.__llm_backend__ != 'pt': raise NotImplementedError(f'Adapter is not supported with {self.llm.__llm_backend__}.')
    config = self.llm.config.model_construct_env(**structured.llm_config)
    stop_token_ids = structured.stop_token_ids or []
    if (config_eos := config['eos_token_id']) and config_eos not in stop_token_ids: stop_token_ids.append(config_eos)
    if self.llm.tokenizer.eos_token_id not in stop_token_ids: stop_token_ids.append(self.llm.tokenizer.eos_token_id)
    if structured.stop is None:
      stop = set()
    elif isinstance(structured.stop, str):
      stop = {structured.stop}
    else:
      stop = set(structured.stop)
    for tid in stop_token_ids:
      if tid: stop.add(self.llm.tokenizer.decode(tid))

    if structured.prompt_token_ids is None:
      if structured.prompt is None: raise ValueError('Either prompt or prompt_token_ids must be specified.')
      prompt_token_ids = self.llm.tokenizer.encode(structured.prompt)
    else:
      prompt_token_ids = structured.prompt_token_ids
    request_id = gen_random_uuid() if structured.request_id is None else structured.request_id

    previous_texts, previous_num_tokens = [''] * config['n'], [0] * config['n']

    try:
      generator = self.runner.generate_iterator(
        prompt_token_ids, request_id, stop=list(stop), adapter_name=structured.adapter_name, **config.model_dump(flatten=True)
      )
    except Exception as err:
      raise RuntimeError(f'Failed to start generation task: {err}') from err

    try:
      async for generated in generator:
        generated = generated.with_options(prompt=structured.prompt)
        delta_outputs = [None] * len(generated.outputs)
        for output in generated.outputs:
          i = output.index
          delta_tokens, delta_text = output.token_ids[previous_num_tokens[i]:], output.text[len(previous_texts[i]):]
          previous_texts[i], previous_num_tokens[i] = output.text, len(output.token_ids)
          delta_outputs[i] = output.with_options(text=delta_text, token_ids=delta_tokens)
        yield f'data: {generated.with_options(outputs=delta_outputs).model_dump_json()}\n\n'
      yield 'data: [DONE]\n\n'
    except Exception as err:
      raise RuntimeError(f'Exception caught during generation: {err}') from err

  inner_cls = types.new_class(
    llm.config.__class__.__name__[:-6] + 'Service',
    (LLMService,),
    exec_body=lambda ns: ns.update({
      'llm': llm,
      'runner': depends(llm.runner),
      'generate_v1': generate_v1,
      'generate_stream_v1': generate_stream_v1,
      '__module__': __name__,
    }),
  )
  svc = types.new_class(
    llm.config.__class__.__name__[:-6] + 'Service',
    (Service,),
    exec_body=lambda ns: ns.update({
      'name': property(lambda self: f"llm-{llm.config['start_name']}-service"),
      '__module__': __name__,
    })
  )(config=validate(attrs), inner=inner_cls)
  if (svc_qualname := svc.__class__.__qualname__) not in (svc_mod := sys.modules[__name__].__dict__): svc_mod[svc_qualname] = svc
  return openllm.mount_entrypoints(svc, llm)


class MessagesConverterInput(t.TypedDict):
  add_generation_prompt: bool
  messages: t.List[t.Dict[str, t.Any]]

class LLMService:
  llm: openllm.LLM[M, T]
  runner: Dependency[Runner[M, T]]

  @openllm.utils.api(output=openllm.MetadataOutput, route='/v1/metadata')
  def metadata_v1(self) -> openllm.MetadataOutput:
    return openllm.MetadataOutput(
      timeout=self.llm.config['timeout'],
      model_name=self.llm.config['model_name'],
      backend=self.llm.__llm_backend__,
      model_id=self.llm.model_id,
      configuration=self.llm.config.model_dump_json().decode(),
    )

  @openllm.utils.api(route='/v1/helpers/messages', input=MessagesConverterInput)
  def helpers_messages_v1(self, message: MessagesConverterInput = MessagesConverterInput(
      add_generation_prompt=False,
      messages=[
        MessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
        MessageParam(role='user', content='Hi there!'),
        MessageParam(role='assistant', content='Yes?'),  #
      ],
  )) -> str:
    add_generation_prompt, messages = message['add_generation_prompt'], message['messages']
    return self.llm.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
