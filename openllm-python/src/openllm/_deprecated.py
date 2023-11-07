from __future__ import annotations
import os
import typing as t
import warnings

import openllm

from openllm_core._typing_compat import LiteralBackend
from openllm_core.utils import first_not_none
from openllm_core.utils import is_vllm_available

if t.TYPE_CHECKING:
  from openllm_core import LLMConfig
  from openllm_core._typing_compat import ParamSpec

  from ._llm import LLMRunner
  P = ParamSpec('P')

_object_setattr = object.__setattr__

def _mark_deprecated(fn: t.Callable[P, t.Any]) -> t.Callable[P, t.Any]:
  _object_setattr(fn, '__deprecated__', True)
  return fn

@_mark_deprecated
def Runner(model_name: str,
           ensure_available: bool = False,
           init_local: bool = False,
           backend: LiteralBackend | None = None,
           llm_config: LLMConfig | None = None,
           **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  '''Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'.

  > [!WARNING]
  > This method is now deprecated and in favor of 'openllm.LLM.runner'

  ```python
  runner = openllm.Runner("dolly-v2")

  @svc.on_startup
  def download():
    runner.download_model()
  ```

  if `init_local=True` (For development workflow), it will also enable `ensure_available`.
  Default value of `ensure_available` is None. If set then use that given value, otherwise fallback to the aforementioned behaviour.

  Args:
    model_name: Supported model name from 'openllm models'
    ensure_available: If True, it will download the model if it is not available. If False, it will skip downloading the model.
                      If False, make sure the model is available locally.
    backend: The given Runner implementation one choose for this Runner. If `OPENLLM_BACKEND` is set, it will respect it.
    llm_config: Optional ``openllm.LLMConfig`` to initialise this ``openllm.LLMRunner``.
    init_local: If True, it will initialize the model locally. This is useful if you want to run the model locally. (Symmetrical to bentoml.Runner.init_local())
    **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs behaviour
  '''
  from ._llm import LLM
  if llm_config is None: llm_config = openllm.AutoConfig.for_model(model_name)
  model_id = attrs.get('model_id') or llm_config['env']['model_id_value']
  _RUNNER_MSG = f'''\
  Using 'openllm.Runner' is now deprecated. Make sure to switch to the following syntax:

  ```python
  llm = openllm.LLM('{model_id}')

  svc = bentoml.Service('...', runners=[llm.runner])

  @svc.api(...)
  async def chat(input: str) -> str:
    async for it in llm.generate_iterator(input): print(it)
  ```
    '''
  warnings.warn(_RUNNER_MSG, DeprecationWarning, stacklevel=2)
  attrs.update({
      'model_id': model_id,
      'quantize': llm_config['env']['quantize_value'],
      'serialisation': first_not_none(attrs.get('serialisation'), os.environ.get('OPENLLM_SERIALIZATION'), default=llm_config['serialisation']),
      'system_message': first_not_none(os.environ.get('OPENLLM_SYSTEM_MESSAGE'), attrs.get('system_message'), None),
      'prompt_template': first_not_none(os.environ.get('OPENLLM_PROMPT_TEMPLATE'), attrs.get('prompt_template'), None),
  })

  backend = t.cast(LiteralBackend, first_not_none(backend, default='vllm' if is_vllm_available() else 'pt'))
  if init_local: ensure_available = True
  llm = LLM[t.Any, t.Any](backend=backend, llm_config=llm_config, **attrs)
  if ensure_available: llm.save_pretrained()
  if init_local: llm.runner.init_local(quiet=True)
  return llm.runner

_DEPRECATED = {k: v for k, v in locals().items() if getattr(v, '__deprecated__', False)}

def __dir__() -> list[str]:
  return sorted(_DEPRECATED.keys())
