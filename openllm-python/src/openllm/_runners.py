from __future__ import annotations
import traceback
import typing as t

import vllm

import bentoml

from openllm.exceptions import OpenLLMException
from openllm_core._typing_compat import LiteralString
from openllm_core.utils import device_count
from openllm_core.utils import get_debug_mode

class vLLMRunnable(bentoml.Runnable):
  SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'amd.com/gpu', 'cpu')
  SUPPORTS_CPU_MULTI_THREADING = True

  def __init__(self, bentomodel: bentoml.Model, tokenizer: LiteralString = 'local') -> None:
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
  async def generate_iterator(self, prompt_token_ids: list[int], **attrs: t.Any) -> t.AsyncGenerator[vllm.RequestOutput, None]:
    request_id: str = attrs.pop('request_id', None)
    if request_id is None: raise ValueError("'request_id' must not be None.")

    stop: set[str] = set()
    stop_var: str | t.Iterable[str] | None = attrs.pop('stop', None)
    if isinstance(stop_var, str) and stop_var != '': stop.add(stop_var)
    elif isinstance(stop_var, t.Iterable): stop.update(stop_var)

    temperature = attrs.pop('temperature', self.config['temperature'])
    top_p = attrs.pop('top_p', self.config['top_p'])

    if temperature <= 1e-5: top_p = 1.0
    sampling_params = self.config.model_construct_env(stop=list(stop), temperature=temperature, top_p=top_p, **attrs).to_sampling_config()
    # async for request_output in self.model.generate(None, sampling_params, request_id, prompt_token_ids): yield f"event: message\ndata: {unmarshal_vllm_outputs(request_output)}\n\n"
    # yield "event: end\n\n"
    async for request_output in self.model.generate(None, sampling_params, request_id, prompt_token_ids):
      yield request_output
