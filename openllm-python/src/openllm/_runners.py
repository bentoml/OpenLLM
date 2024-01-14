from __future__ import annotations
import gc, types, inspect, typing as t
import bentoml, openllm
from openllm_core.exceptions import OpenLLMException, MissingDependencyError
from openllm_core._schemas import CompletionChunk, GenerationOutput, SampleLogprobs
from openllm_core._typing_compat import LiteralString, ParamSpec, overload
from openllm_core.utils import ReprMixin, is_ctranslate_available, is_vllm_available, correct_closure

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import M, T
  from ._runners import Runner

P, R = ParamSpec('P'), t.TypeVar('R')

_registry = {}
__all__ = ['runner']

@overload
def registry(cls: None = ..., /, *, alias: LiteralString = ...) -> type[Runner[M, T]]: ...
@overload
def registry(cls: type[Runner[M, T]], /) -> type[Runner[M, T]]: ...
def registry(cls: type[Runner[M, T]] | None = None, *, alias: LiteralString | None = None) -> type[Runner[M, T]]:
  def _gated_new(cls: type[Runner[M, T]], /, **_: t.Any) -> Runner[M, T]:
    if not cls.__openllm_can_init__:
      raise TypeError(f'Cannot instantiate {cls.__name__} directly.')
    return object.__new__(cls)

  def decorator(_cls: type[Runner[M, T]]) -> type[Runner[M, T]]:
    _registry[_cls.__name__[:-8].lower() if alias is None else alias] = _cls
    cd = {'__openllm_can_init__': False, '__new__': _gated_new}
    cd.update(dict(_cls.__dict__))
    cls = type(_cls)(_cls.__name__, _cls.__bases__, cd)
    return correct_closure(cls, _cls)

  return decorator(cls) if cls is not None else decorator

def runner(llm):
  try:
    assert llm.bentomodel
  except (bentoml.exceptions.NotFound, AssertionError) as err:
    raise RuntimeError(f'Failed to locate {llm.bentomodel}: {err}') from err

  def make_runnable_caller(method):
    async def _runnable_to_sse(self, *args, **attrs):
      async for it in method(self, *args, **attrs): yield f"data: {it.model_dump_json()}\n\n"
    return _runnable_to_sse

  return types.new_class(
    f'{(classname := llm.config.__class__.__name__[:-6])}Runner',
    (bentoml.Runner,),
    exec_body=lambda ns: ns.update({
      'llm_type': llm.llm_type,
      'identifying_params': llm.identifying_params,  #
      'llm_tag': llm.tag,
      'llm': llm,
      'config': llm.config,
      'backend': llm.__llm_backend__,  #
      '__module__': llm.__module__,
      '__repr__': ReprMixin.__repr__,  #
      '__openllm_can_init__': True,
      '__doc__': llm.config.__class__.__doc__ or f'Generated Runner class for {llm.config["model_name"]}',
      '__repr_keys__': property(lambda _: {'config', 'llm_type', 'runner_methods', 'backend', 'llm_tag'}),
      '__repr_args__': lambda _: (
        (
          'runner_methods',
          {
            method.name: {'batchable': method.config.batchable, 'batch_dim': method.config.batch_dim if method.config.batchable else None}
            for method in _.runner_methods
          },
        ),
        ('config', llm.config.model_dump(flatten=True)),
        ('llm_type', llm.llm_type),
        ('backend', llm.__llm_backend__),
        ('llm_tag', llm.tag),
      ),
      'has_adapters': llm.has_adapters,
      'template': llm.config.template,
      'system_message': llm.config.system_message,
    }),
  )(
    types.new_class(
      f'{(backend_cls := _registry[llm.__llm_backend__]).__name__[:-8]}{classname}Runnable',
      (bentoml.Runnable,),
      exec_body=lambda ns: ns.update({
        'SUPPORTED_RESOURCES': ('nvidia.com/gpu', 'amd.com/gpu', 'cpu'),
        'SUPPORTS_CPU_MULTI_THREADING': True,
        '__module__': __name__,
        'llm': llm,
        **{
          name: bentoml.Runnable.method(batchable=False)(make_runnable_caller(api.func))
          for name, api in vars(backend_cls).items()
          if hasattr(api, 'func') and (mod := inspect.getmodule(api)) is not None and mod.__name__ == '_bentoml_sdk.api'
        },  # XXX: hack for cases when BentoML > 1.2 is not installed?
      }),
    ),
    name=f"llm-{llm.config['start_name']}-runner",
    models=[llm.bentomodel],
    scheduling_strategy=openllm.CascadingResourceStrategy,
  )

def bases(llm):
  try:
    assert llm.bentomodel
  except (bentoml.exceptions.NotFound, AssertionError) as err:
    raise RuntimeError(f'Failed to locate {llm.bentomodel}: {err}') from err

  return types.new_class(
    (backend_cls := _registry[llm.__llm_backend__]).__name__[:-8] + llm.config.__class__.__name__[:-6] + 'RunnerService',
    (backend_cls,),
    exec_body=lambda ns: ns.update({
      'llm': llm,
      'llm_tag': llm.tag,
      'llm_config': llm.config,
      'llm_bentomodel': llm.bentomodel,
      'llm_type': llm.llm_type,
      'backend': llm.__llm_backend__,
      'identifying_params': llm.identifying_params,
      '__module__': __name__,
      '__openllm_can_init__': True,
      '__repr__': ReprMixin.__repr__,
      '__doc__': llm.config.__class__.__doc__ or f'Generated RunnerService for {llm.config["model_name"]}',
      '__repr_keys__': property(lambda _: {'llm_config', 'llm_type', 'llm_tag', 'backend'}),
      '__repr_args__': lambda _: (
        ('llm_config', llm.config.model_dump(flatten=True)),
        ('llm_type', llm.llm_type),
        ('llm_tag', llm.tag),
        ('backend', llm.__llm_backend__),
      ),
      'has_adapters': llm.has_adapters,
      'template': llm.config.template,
      'system_message': llm.config.system_message,
      'impl': backend_cls,
    }),
  )


@registry
class vLLMRunnable:
  llm: openllm.LLM

  def __init__(self) -> None:
    if not is_vllm_available():
      raise MissingDependencyError('vLLM is not installed. Do `pip install "openllm[vllm]"`.')
    self.model, self.tokenizer = self.llm.model, self.llm.tokenizer

  @openllm.utils.api(output=GenerationOutput)
  async def generate_iterator(
    self,
    prompt_token_ids: t.List[int],
    request_id: str,
    stop: t.Optional[t.Iterable[str]] = None,
    adapter_name: t.Optional[str] = None,
    **attrs: t.Any,
  ) -> t.AsyncGenerator[GenerationOutput, None]:
    _, sampling_params = self.llm.config.model_construct_env(stop=list(stop) if stop else None, **attrs).inference_options(self.llm)
    async for request_output in self.model.generate(None, sampling_params, request_id, prompt_token_ids):
      yield GenerationOutput.from_vllm(request_output).model_dump_json()


@registry(alias='pt')
class PyTorchRunnable:
  llm: openllm.LLM

  def __init__(self):
    import torch

    self.model, self.tokenizer = self.llm.model, self.llm.tokenizer
    self.is_encoder_decoder = self.llm.model.config.is_encoder_decoder
    self.device = self.llm.model.device if hasattr(self.llm.model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  @openllm.utils.api(output=GenerationOutput)
  async def generate_iterator(
    self,
    prompt_token_ids: t.List[int],
    request_id: str,
    stop: t.Optional[t.Iterable[str]] = None,
    adapter_name: t.Optional[str] = None,
    **attrs: t.Any,
  ) -> t.AsyncGenerator[GenerationOutput, None]:
    from ._generation import get_context_length, prepare_logits_processor
    import torch

    if adapter_name is not None:
      self.model.set_adapter(adapter_name)

    max_new_tokens = attrs.pop('max_new_tokens', 256)
    context_length = attrs.pop('context_length', None)
    if context_length is None:
      context_length = get_context_length(self.model.config)
    if self.model.config.is_encoder_decoder:
      max_src_len = context_length
    else:
      max_src_len = context_length - max_new_tokens - 1
    prompt_token_ids = prompt_token_ids[-max_src_len:]

    stop_token_ids = [self.tokenizer.encode(it) for it in stop] if stop else []
    if self.tokenizer.eos_token_id not in stop_token_ids:  # add eos token
      stop_token_ids.append(self.tokenizer.eos_token_id)

    config = self.llm.config.model_construct_env(max_new_tokens=max_new_tokens, **attrs)
    logits_processor = prepare_logits_processor(config)
    cumulative_logprob = 0.0

    with torch.inference_mode():
      output_token_ids = list(prompt_token_ids)
      input_len = len(prompt_token_ids)

      if self.is_encoder_decoder:
        if config['logprobs']:  # FIXME: logprobs is not supported
          raise NotImplementedError('Logprobs is yet to be supported with encoder-decoder models.')
        encoder_output = self.model.encoder(input_ids=torch.as_tensor([prompt_token_ids], device=self.device))[0]
        start_ids = torch.as_tensor([[self.model.generation_config.decoder_start_token_id]], dtype=torch.int64, device=self.device)
      else:
        start_ids = torch.as_tensor([prompt_token_ids], device=self.device)

      past_key_values = out = token = None
      finish_reason = None
      prompt_logprobs = []
      prompt_token_indices = []
      stopped = False
      sample_logprobs: SampleLogprobs = [None]  # The first token has no logprobs

      for i in range(config['max_new_tokens']):
        if i == 0:  # prefill
          if self.is_encoder_decoder:
            out = self.model.decoder(input_ids=start_ids, encoder_hidden_states=encoder_output, use_cache=True)
            logits = self.model.lm_head(out[0])
          else:
            out = self.model(input_ids=start_ids, use_cache=True)
            logits = out.logits
        elif self.is_encoder_decoder:  # decoding
          out = self.model.decoder(
            input_ids=torch.as_tensor([[token]], device=self.device),
            encoder_hidden_states=encoder_output,
            past_key_values=past_key_values,
            use_cache=True,
          )
          logits = self.model.lm_head(out[0])
        else:
          out = self.model(input_ids=torch.as_tensor([[token]], device=self.device), past_key_values=past_key_values, use_cache=True)
          logits = out.logits
        past_key_values = out.past_key_values
        if logits_processor:
          if config['repetition_penalty'] > 1.0:
            tmp_output_ids: t.Any = torch.as_tensor([output_token_ids], device=self.device)
          else:
            tmp_output_ids = None
          last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
          last_token_logits = logits[0, -1, :]

        # Switch to CPU by avoiding some bugs in mps backend.
        if self.device.type == 'mps':
          last_token_logits = last_token_logits.float().to('cpu')

        # TODO: refactor for better sampling logic and apply penalties correctly
        # support sequence generation, best_of
        if config['temperature'] < 1e-5 or config['top_p'] < 1e-8:  # greedy
          _, indices = torch.topk(last_token_logits, 2)
        else:
          probs = torch.softmax(last_token_logits, dim=-1, dtype=torch.float)
          indices = torch.multinomial(probs, num_samples=2)
        tokens = [int(token) for token in indices.tolist()]

        token = tokens[0]
        output_token_ids.append(token)
        if config['logprobs']:
          # NOTE: We can't use last_token_logits since logprobs is based on raw logits
          logprobs = torch.log_softmax(logits[0, -1, :], dim=-1, dtype=torch.float)
          token_logprobs = logprobs[token].item()
          cumulative_logprob += token_logprobs

        if config['prompt_logprobs']:
          for token_id in prompt_token_ids:
            if token_id in prompt_token_indices:
              continue
            prompt_token_indices.append(token_id)
            prompt_logprobs.append({token_id: logprobs[token_id].item()})

        stopped = token in stop_token_ids

        tmp_output_ids, rfind_start = output_token_ids[input_len:], 0
        # XXX: Move this to API server
        text = self.tokenizer.decode(tmp_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)

        if len(stop) > 0:
          for it in stop:
            pos = text.rfind(it, rfind_start)
            if pos != -1:
              text, stopped = text[:pos], True
              break

        if config['logprobs']:
          sample_logprobs.append({token: token_logprobs})

        yield GenerationOutput(
          prompt='',
          finished=False,
          outputs=[
            CompletionChunk(
              index=0,
              text=text,
              token_ids=tmp_output_ids,
              cumulative_logprob=cumulative_logprob,
              logprobs=sample_logprobs if config['logprobs'] else None,
              finish_reason=None,
            )
          ],
          prompt_token_ids=prompt_token_ids,
          prompt_logprobs=prompt_logprobs if config['prompt_logprobs'] else None,
          request_id=request_id,
        ).model_dump_json()
        if stopped:
          break
      else:
        finish_reason = 'length'
      if stopped:
        finish_reason = 'stop'
      yield GenerationOutput(
        prompt='',
        finished=True,
        outputs=[
          CompletionChunk(
            index=0,
            text=text,
            token_ids=output_token_ids,
            cumulative_logprob=cumulative_logprob,
            logprobs=sample_logprobs if config['logprobs'] else None,
            finish_reason=finish_reason,
          )
        ],
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=prompt_logprobs if config['prompt_logprobs'] else None,
        request_id=request_id,
      ).model_dump_json()

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@registry
class CTranslateRunnable:
  llm: openllm.LLM

  def __init__(self) -> None:
    if not is_ctranslate_available():
      raise MissingDependencyError('ctranslate is not installed. Do `pip install "openllm[ctranslate]"`')
    self.model, self.tokenizer = self.llm.model, self.llm.tokenizer

  @openllm.utils.api(output=GenerationOutput)
  async def generate_iterator(
    self,
    prompt_token_ids: t.List[int],
    request_id: str,
    stop: t.Optional[t.Iterable[str]] = None,
    adapter_name: t.Optional[str] = None,
    **attrs: t.Any,
  ) -> t.AsyncGenerator[GenerationOutput, None]:
    if adapter_name is not None:
      raise OpenLLMException('Adapter is not supported with CTranslate')
    config, sampling_params = self.llm.config.model_construct_env(stop=list(stop) if stop else None, **attrs).inference_options(self.llm)
    cumulative_logprob, output_token_ids, input_len = 0.0, list(prompt_token_ids), len(prompt_token_ids)
    tokens = self.tokenizer.convert_ids_to_tokens(prompt_token_ids)
    async for request_output in self.model.async_generate_tokens(tokens, **sampling_params):
      if config['logprobs']:
        cumulative_logprob += request_output.log_prob
      output_token_ids.append(request_output.token_id)
      text = self.tokenizer.decode(
        output_token_ids[input_len:], skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True
      )
      yield GenerationOutput(
        prompt_token_ids=prompt_token_ids,
        prompt='',
        finished=request_output.is_last,
        request_id=request_id,
        outputs=[
          CompletionChunk(
            index=0,
            text=text,
            finish_reason=None,
            token_ids=output_token_ids[input_len:],
            cumulative_logprob=cumulative_logprob,
            # TODO: logprobs, but seems like we don't have access to the raw logits
          )
        ],
      ).model_dump_json()
