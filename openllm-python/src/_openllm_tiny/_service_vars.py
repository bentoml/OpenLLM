import orjson, openllm_core.utils as coreutils

(
  model_id,
  model_name,
  quantise,
  serialisation,
  dtype,
  trust_remote_code,
  max_model_len,
  gpu_memory_utilization,
  services_config,
) = (
  coreutils.getenv('model_id', var=['MODEL_ID']),
  coreutils.getenv('model_name'),
  coreutils.getenv('quantize', var=['QUANTISE']),
  coreutils.getenv('serialization', default='safetensors', var=['SERIALISATION']),
  coreutils.getenv('dtype', default='auto', var=['TORCH_DTYPE']),
  coreutils.check_bool_env('TRUST_REMOTE_CODE', False),
  orjson.loads(coreutils.getenv('max_model_len', default=orjson.dumps(None))),
  orjson.loads(coreutils.getenv('gpu_memory_utilization', default=orjson.dumps(0.9), var=['GPU_MEMORY_UTILISATION'])),
  orjson.loads(coreutils.getenv('services_config', orjson.dumps({}))),
)
