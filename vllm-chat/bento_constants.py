CONSTANT_YAML = """
alias:
  - latest
  - 3.8b
  - instruct
  - mini
template: vllm-chat
service_config:
  name: phi3
  traffic:
    timeout: 300
  resources:
    gpu: 1
    gpu_type: nvidia-tesla-t4
prompt:
  head: ~
  body: |-
    <|user|>
    {user_prompt}<|end|>
    <|assistant|>
chat_template: phi-3
engine_config:
  model: microsoft/Phi-3-mini-4k-instruct
  max_model_len: 4096
  dtype: half
  trust_remote_code: true
"""
