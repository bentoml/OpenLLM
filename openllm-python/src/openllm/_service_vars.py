from __future__ import annotations
import os


model_id = os.environ['OPENLLM_MODEL_ID']  # openllm: model name
model_tag = None  # openllm: model tag
adapter_map = os.environ['OPENLLM_ADAPTER_MAP']  # openllm: model adapter map
