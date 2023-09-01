'''LLM assignment magik.'''
from __future__ import annotations
import functools
import math
import traceback
import typing as t

import openllm

from openllm.exceptions import OpenLLMException
from openllm_core._configuration import _object_getattribute
from openllm_core._configuration import _setattr_class
from openllm_core._schema import unmarshal_vllm_outputs
from openllm_core._typing_compat import DictStrAny
from openllm_core._typing_compat import ListStr
from openllm_core._typing_compat import M
from openllm_core._typing_compat import T
from openllm_core._typing_compat import import_model_protocol
from openllm_core._typing_compat import llm_post_init_protocol
from openllm_core._typing_compat import load_model_protocol
from openllm_core._typing_compat import load_tokenizer_protocol
from openllm_core.utils import LazyLoader
from openllm_core.utils import codegen
from openllm_core.utils import device_count
from openllm_core.utils import first_not_none
from openllm_core.utils import is_torch_available

if t.TYPE_CHECKING:
  import torch
  import vllm

  import bentoml

  from openllm._llm import LLM
else:
  torch = LazyLoader('torch', globals(), 'torch')
  vllm = LazyLoader('vllm', globals(), 'vllm')

def import_model(fn: import_model_protocol[bentoml.Model, M, T]) -> t.Callable[[LLM[M, T]], bentoml.Model]:
  @functools.wraps(fn)
  def inner(self: LLM[M, T], *decls: t.Any, trust_remote_code: bool | None = None, **attrs: t.Any) -> bentoml.Model:
    trust_remote_code = first_not_none(trust_remote_code, default=self.trust_remote_code)
    (model_decls, model_attrs), _ = self.llm_parameters
    decls = (*model_decls, *decls)
    attrs = {**model_attrs, **attrs}
    return fn(self, *decls, trust_remote_code=trust_remote_code, **attrs)

  return inner

def load_model(fn: load_model_protocol[M, T]) -> t.Callable[[LLM[M, T]], M | vllm.LLMEngine]:
  @functools.wraps(fn)
  def inner(self: LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M | vllm.LLMEngine:
    if self.__llm_backend__ == 'vllm':
      num_gpus, dev = 1, device_count()
      if dev >= 2: num_gpus = dev if dev // 2 == 0 else math.ceil(dev / 2)
      # TODO: Do some more processing with token_id once we support token streaming
      try:
        return vllm.LLMEngine.from_engine_args(
            vllm.EngineArgs(model=self._bentomodel.path,
                            tokenizer=self._bentomodel.path if self.tokenizer_id == 'local' else self.tokenizer_id,
                            tokenizer_mode='auto',
                            tensor_parallel_size=num_gpus,
                            dtype='auto',
                            worker_use_ray=False))
      except Exception as err:
        traceback.print_exc()
        raise OpenLLMException(f'Failed to initialise vLLMEngine due to the following error:\n{err}') from None
    else:
      (model_decls, model_attrs), _ = self.llm_parameters
      return fn(self, *(*model_decls, *decls), **{**model_attrs, **attrs})

  return inner

def load_tokenizer(fn: load_tokenizer_protocol[M, T]) -> t.Callable[[LLM[M, T]], T]:
  @functools.wraps(fn)
  def inner(self: LLM[M, T], **tokenizer_attrs: t.Any) -> T:
    return fn(self, **{**self.llm_parameters[-1], **tokenizer_attrs})

  return inner

def llm_post_init(fn: llm_post_init_protocol[M, T]) -> t.Callable[[LLM[M, T]], None]:
  @functools.wraps(fn)
  def inner(self: LLM[M, T]) -> None:
    if self.__llm_backend__ == 'pt' and is_torch_available():
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn(self)

  return inner

def make_llm_attributes(cls: type[LLM[M, T]]) -> t.Callable[[type[LLM[M, T]]], None]:
  '''Make LLM attributes for the given LLM subclass.'''
  from ._llm import LLM
  from ._llm import LLMFunction
  from ._llm import LLMInterface
  from ._llm import LLMSerialisation

  args: ListStr = []
  globs: DictStrAny = {'cls': cls, '__wrapped_llm_post_init': llm_post_init, 'LLM': LLM}
  # _cached_LLMFunction_get and _ccached_LLMSerialisation_get
  globs.update({f'_cached_{cl_.__name__}_get': _object_getattribute.__get__(cl_) for cl_ in {LLMSerialisation, LLMFunction}})
  # llm_post_init implementation
  lines: ListStr = [f'_impl_{cls.__name__}_func=cls.llm_post_init', _setattr_class('llm_post_init', f'__wrapped_llm_post_init(_impl_{cls.__name__}_func)')]

  serialisation_attr = {'import_model': import_model, 'load_model': load_model, 'load_tokenizer': load_tokenizer,}
  for func, impl in serialisation_attr.items():
    impl_name = f'__wrapped_{func}'
    globs.update({f'__serialisation_{func}': getattr(openllm.serialisation, func, None), impl_name: impl})
    cached_func_name = f'_cached_{cls.__name__}_func'
    func_call = f"_impl_{cls.__name__}_{func}={cached_func_name} if {cached_func_name} is not _cached_LLMSerialisation_get('{func}') else __serialisation_{func}"
    lines.extend([f'{cached_func_name}=cls.{func}', func_call, _setattr_class(func, f'{impl_name}(_impl_{cls.__name__}_{func})')])

  # assign vLLM implementation
  if cls.__llm_backend__ == 'vllm':
    vllm_func = {f'_vllm_{it}': fn for it, fn in zip(('generate', 'generate_iterator', 'postprocess_generate'), (vllm_generate, vllm_generate_iterator, vllm_postprocess_generate))}
    globs.update(vllm_func)
    lines.extend([_setattr_class(it[6:], it) for it in vllm_func])

  interface_anns = codegen.get_annotations(LLMInterface)

  # cached attribute initialisation
  def dunder_cached(key: str) -> str:
    return f'__llm_{key}__'

  st_attr = {'model', 'tokenizer', 'adapter_map'}
  lines.extend([_setattr_class(dunder_cached(v), None) for v in st_attr])

  # boolean for better LLM implementation resolver
  def dunder_support(key: str) -> str:
    return f'__llm_supports_{key}__'

  bool_attr = {it[15:-2] for it in interface_anns if it.startswith('__llm_supports_')}
  lines.extend([_setattr_class(dunder_support(fn), f"cls.{fn} is not _cached_LLMFunction_get('{fn}')") for fn in bool_attr])

  return codegen.generate_function(cls, '__assign_llm_attr', lines, args=('cls', *args), globs=globs, annotations={'cls': 't.Type[LLM]', 'return': None})

def vllm_postprocess_generate(self: LLM['vllm.LLMEngine', T], prompt: str, generation_result: list[dict[str, t.Any]], **_: t.Any) -> str:
  return generation_result[0]['outputs'][0]['text']

def vllm_generate_iterator(self: LLM['vllm.LLMEngine', T],
                           prompt: str,
                           /,
                           *,
                           echo: bool = False,
                           stop: str | t.Iterable[str] | None = None,
                           stop_token_ids: list[int] | None = None,
                           **attrs: t.Any) -> t.Iterator[dict[str, t.Any]]:
  request_id: str | None = attrs.pop('request_id', None)
  if request_id is None: raise ValueError('request_id must not be None.')
  if stop_token_ids is None: stop_token_ids = []
  stop_token_ids.append(self.tokenizer.eos_token_id)
  stop_: set[str] = set()
  if isinstance(stop, str) and stop != '': stop_.add(stop)
  elif isinstance(stop, list) and stop != []: stop_.update(stop)
  for tid in stop_token_ids:
    if tid: stop_.add(self.tokenizer.decode(tid))

  if self.config['temperature'] <= 1e-5: top_p = 1.0
  else: top_p = self.config['top_p']
  config = self.config.model_construct_env(stop=list(stop_), top_p=top_p, **attrs)
  self.model.add_request(request_id=request_id, prompt=prompt, sampling_params=config.to_sampling_config())
  while self.model.has_unfinished_requests():
    for request_output in self.model.step():
      prompt = request_output.prompt
      if echo: text_outputs = [prompt + output.text for output in request_output.outputs]
      else: text_outputs = [output.text for output in request_output.outputs]
      yield {'text': text_outputs, 'error_code': 0}
      if request_output.finished: break

def vllm_generate(self: LLM['vllm.LLMEngine', T], prompt: str, **attrs: t.Any) -> list[dict[str, t.Any]]:
  request_id: str | None = attrs.pop('request_id', None)
  if request_id is None: raise ValueError('request_id must not be None.')
  outputs: list[vllm.RequestOutput] = []
  # TODO: support prompt_token_ids
  self.model.add_request(request_id=request_id, prompt=prompt, sampling_params=self.config.model_construct_env(**attrs).to_sampling_config())
  while self.model.has_unfinished_requests():
    outputs.extend([r for r in self.model.step() if r.finished])
  return [unmarshal_vllm_outputs(i) for i in outputs]
