import orjson, typing as t, openllm_core.utils as coreutils
from openllm_core._typing_compat import LiteralSerialisation, LiteralQuantise
from _openllm_tiny._llm import Dtype

(
  model_id,
  model_tag,
  model_version,
  quantise,
  serialisation,
  dtype,
  trust_remote_code,
  max_model_len,
  gpu_memory_utilization,
  services_config,
) = (
  coreutils.getenv('model_id', var=['MODEL_ID'], return_type=str),
  coreutils.getenv('model_tag', default=None, return_type=t.Optional[str]),
  coreutils.getenv('model_version', default=None, return_type=t.Optional[str]),
  coreutils.getenv('quantize', var=['QUANTISE'], return_type=LiteralQuantise),
  coreutils.getenv('serialization', default='safetensors', var=['SERIALISATION'], return_type=LiteralSerialisation),
  coreutils.getenv('dtype', default='auto', var=['TORCH_DTYPE'], return_type=Dtype),
  coreutils.check_bool_env('TRUST_REMOTE_CODE', False),
  t.cast(t.Optional[int], orjson.loads(coreutils.getenv('max_model_len', default=orjson.dumps(None)))),
  t.cast(
    float,
    orjson.loads(
      coreutils.getenv('gpu_memory_utilization', default=orjson.dumps(0.9), var=['GPU_MEMORY_UTILISATION'])
    ),
  ),
  t.cast(t.Dict[str, t.Any], orjson.loads(coreutils.getenv('services_config', orjson.dumps({})))),
)
