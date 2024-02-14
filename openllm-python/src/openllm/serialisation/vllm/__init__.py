import openllm, traceback
from openllm_core.utils import is_vllm_available
from ..transformers import import_model

__all__ = ['import_model', 'load_model']


def load_model(llm, *decls, **attrs):
  if not is_vllm_available():
    raise RuntimeError(
      "'vllm' is required to use with backend 'vllm'. Install it with 'pip install \"openllm[vllm]\"'"
    )
  import vllm, torch

  num_gpus, dev = 1, openllm.utils.device_count()
  if dev >= 2:
    num_gpus = min(dev // 2 * 2, dev)
  quantise = llm.quantise if llm.quantise and llm.quantise in {'gptq', 'awq', 'squeezellm'} else None
  dtype = (
    torch.float16 if quantise == 'gptq' else llm._torch_dtype
  )  # NOTE: quantise GPTQ doesn't support bfloat16 yet.
  try:
    return vllm.AsyncLLMEngine.from_engine_args(
      vllm.AsyncEngineArgs(
        worker_use_ray=False,
        engine_use_ray=False,
        tokenizer_mode='auto',
        tensor_parallel_size=num_gpus,
        model=llm.bentomodel.path,
        tokenizer=llm.bentomodel.path,
        trust_remote_code=llm.trust_remote_code,
        dtype=dtype,
        max_model_len=llm._max_model_len,
        gpu_memory_utilization=llm._gpu_memory_utilization,
        quantization=quantise,
      )
    )
  except Exception as err:
    traceback.print_exc()
    raise openllm.exceptions.OpenLLMException(
      f'Failed to initialise vLLMEngine due to the following error:\n{err}'
    ) from err
