import os, orjson, openllm_core.utils as coreutils

model_id, model_tag, adapter_map, serialization, trust_remote_code = (
  os.environ['OPENLLM_MODEL_ID'],
  None,
  orjson.loads(os.getenv('OPENLLM_ADAPTER_MAP', orjson.dumps(None))),
  os.getenv('OPENLLM_SERIALIZATION', default='safetensors'),
  coreutils.check_bool_env('TRUST_REMOTE_CODE', False),
)
max_model_len, gpu_memory_utilization = (
  orjson.loads(os.getenv('MAX_MODEL_LEN', orjson.dumps(None).decode())),
  orjson.loads(os.getenv('GPU_MEMORY_UTILIZATION', orjson.dumps(0.9).decode())),
)
