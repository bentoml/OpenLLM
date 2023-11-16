import os

import orjson

from openllm_core.utils import ENV_VARS_TRUE_VALUES

model_id = os.environ['OPENLLM_MODEL_ID']
model_tag = None
adapter_map = orjson.loads(os.getenv('OPENLLM_ADAPTER_MAP', orjson.dumps(None)))
prompt_template = os.getenv('OPENLLM_PROMPT_TEMPLATE')
system_message = os.getenv('OPENLLM_SYSTEM_MESSAGE')
serialization = os.getenv('OPENLLM_SERIALIZATION', default='safetensors')
trust_remote_code = str(os.getenv('TRUST_REMOTE_CODE', default=str(False))).upper() in ENV_VARS_TRUE_VALUES
