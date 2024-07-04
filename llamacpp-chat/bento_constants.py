CONSTANT_YAML = """
  project: llamacpp-chat
  service_config:
    name: phi3
    traffic:
      timeout: 300
    resources:
      memory: 3Gi
  engine_config:
    model: microsoft/Phi-3-mini-4k-instruct-gguf
    max_model_len: 2048
  chat_template: phi-3
"""
