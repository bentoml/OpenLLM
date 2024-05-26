import orjson, openllm_core.utils as coreutils
from openllm_core._typing_compat import LiteralSerialisation, LiteralQuantise
from _openllm_tiny._llm import Dtype

(
  model_id,
  revision,
  quantise,
  serialisation,
  dtype,
  trust_remote_code,
  max_model_len,
  gpu_memory_utilization,
  services_config,
) = (
  coreutils.getenv('model_id', var=['MODEL_ID'], return_type=str),
  orjson.loads(coreutils.getenv('revision', return_type=str)),
  coreutils.getenv('quantize', var=['QUANTISE'], return_type=LiteralQuantise),
  coreutils.getenv('serialization', default='safetensors', var=['SERIALISATION'], return_type=LiteralSerialisation),
  coreutils.getenv('dtype', default='auto', var=['TORCH_DTYPE'], return_type=Dtype),
  coreutils.check_bool_env('TRUST_REMOTE_CODE', False),
  orjson.loads(coreutils.getenv('max_model_len', default=orjson.dumps(None))),
  orjson.loads(coreutils.getenv('gpu_memory_utilization', default=orjson.dumps(0.9), var=['GPU_MEMORY_UTILISATION'])),
  orjson.loads(coreutils.getenv('services_config', orjson.dumps({}))),
)
