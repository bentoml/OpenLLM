from __future__ import annotations

import typing as t
import bentoml, vllm, transformers, traceback
from openllm.exceptions import OpenLLMException
from bentoml._internal.models.model import ModelSignature
from openllm_core._typing_compat import ModelSignatureDict
from openllm_core.utils import converter
from openllm_core._typing_compat import T
from openllm_core.utils import device_count

if t.TYPE_CHECKING: import transformers

class vLLMRunnable(bentoml.Runnable):
  SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'amd.com/gpu', 'cpu')
  SUPPORTS_CPU_MULTI_THREADING = True

  def __init__(self, bentomodel: bentoml.Model, tokenizer: t.LiteralString = 'local') -> None:
    num_gpus, dev = 1, device_count()
    if dev >= 2: num_gpus = min(dev // 2 * 2, dev)
    try:
      self.model = vllm.AsyncLLMEngine.from_engine_args(
          vllm.AsyncEngineArgs(model=bentomodel.path,
                                tokenizer=bentomodel.path if tokenizer == 'local' else tokenizer,
                                tokenizer_mode='auto',
                                tensor_parallel_size=num_gpus,
                                dtype='auto',
                                disable_log_requests=not get_debug_mode(),
                                worker_use_ray=False,
                                engine_use_ray=False))
    except Exception as err:
      traceback.print_exc()
      raise OpenLLMException(f'Failed to initialise vLLMEngine due to the following error:\n{err}') from err

  @bentoml.Runnable.method(batchable=False)
  def generate_iterator(self, prompt_token_ids: list[int]) -> t.AsyncGenerator[vllm.RequestOutput, None]: ...

