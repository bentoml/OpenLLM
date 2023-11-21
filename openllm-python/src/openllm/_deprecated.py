from __future__ import annotations
import logging
import os
import typing as t
import warnings

import openllm
from openllm_core._typing_compat import LiteralBackend, ParamSpec
from openllm_core.utils import first_not_none, is_vllm_available

P = ParamSpec('P')

logger = logging.getLogger(__name__)


def Runner(
  model_name: str,
  ensure_available: bool = True,
  init_local: bool = False,
  backend: LiteralBackend | None = None,
  llm_config: openllm.LLMConfig | None = None,
  **attrs: t.Any,
) -> openllm.LLMRunner[t.Any, t.Any]:
  """Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'.

  > [!WARNING]
  > This method is now deprecated and in favor of 'openllm.LLM'

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
                      If False, make sure the model is available locally. Default to True, and openllm.LLM will always check if models
                      are available locally. based on generated tag.
    backend: The given Runner implementation one choose for this Runner. If `OPENLLM_BACKEND` is set, it will respect it.
    llm_config: Optional ``openllm.LLMConfig`` to initialise this ``openllm.LLMRunner``.
    init_local: If True, it will initialize the model locally. This is useful if you want to run the model locally. (Symmetrical to bentoml.Runner.init_local())
    **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs behaviour
  """
  from ._llm import LLM

  if llm_config is None:
    llm_config = openllm.AutoConfig.for_model(model_name)
  if not ensure_available:
    logger.warning(
      "'ensure_available=False' won't have any effect as LLM will always check to download the model on initialisation."
    )
  model_id = attrs.get('model_id', os.getenv('OPENLLM_MODEL_ID', llm_config['default_id']))
  _RUNNER_MSG = f"""\
  Using 'openllm.Runner' is now deprecated. Make sure to switch to the following syntax:

  ```python
  llm = openllm.LLM('{model_id}')

  svc = bentoml.Service('...', runners=[llm.runner])

  @svc.api(...)
  async def chat(input: str) -> str:
    async for it in llm.generate_iterator(input): print(it)
  ```
    """
  warnings.warn(_RUNNER_MSG, DeprecationWarning, stacklevel=2)
  attrs.update(
    {
      'model_id': model_id,
      'quantize': os.getenv('OPENLLM_QUANTIZE', attrs.get('quantize', None)),
      'serialisation': first_not_none(
        attrs.get('serialisation'), os.environ.get('OPENLLM_SERIALIZATION'), default=llm_config['serialisation']
      ),
    }
  )

  backend = t.cast(LiteralBackend, first_not_none(backend, default='vllm' if is_vllm_available() else 'pt'))
  llm = LLM[t.Any, t.Any](backend=backend, llm_config=llm_config, embedded=init_local, **attrs)
  return llm.runner


__all__ = ['Runner']
