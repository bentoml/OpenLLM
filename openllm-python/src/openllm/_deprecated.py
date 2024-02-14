from __future__ import annotations
import logging, os, warnings, typing as t
import openllm
from openllm_core._typing_compat import LiteralBackend
from openllm_core.utils import first_not_none, getenv, is_vllm_available

__all__ = ['Runner']
logger = logging.getLogger(__name__)


def Runner(
  model_name: str,
  ensure_available: bool = True,  #
  init_local: bool = False,
  backend: LiteralBackend | None = None,  #
  llm_config: openllm.LLMConfig | None = None,
  **attrs: t.Any,
):
  if llm_config is None:
    llm_config = openllm.AutoConfig.for_model(model_name)
  if not ensure_available:
    logger.warning(
      "'ensure_available=False' won't have any effect as LLM will always check to download the model on initialisation."
    )
  model_id = attrs.get('model_id', os.getenv('OPENLLM_MODEL_ID', llm_config['default_id']))
  warnings.warn(
    f"""\
  Using 'openllm.Runner' is now deprecated. Make sure to switch to the following syntax:

  ```python
  llm = openllm.LLM('{model_id}')

  svc = bentoml.Service('...', runners=[llm.runner])

  @svc.api(...)
  async def chat(input: str) -> str:
    async for it in llm.generate_iterator(input): print(it)
  ```""",
    DeprecationWarning,
    stacklevel=2,
  )
  attrs.update({
    'model_id': model_id,
    'quantize': getenv('QUANTIZE', var=['QUANTISE'], default=attrs.get('quantize', None)),  #
    'serialisation': getenv(
      'serialization', default=attrs.get('serialisation', llm_config['serialisation']), var=['SERIALISATION']
    ),
  })
  # XXX: Make this back to Runnable implementation
  return openllm.LLM(
    backend=first_not_none(backend, default='vllm' if is_vllm_available() else 'pt'),
    llm_config=llm_config,
    embedded=init_local,
    **attrs,
  ).runner
