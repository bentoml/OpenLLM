


PROMPT_TEMPLATE_MAP = {
    "llama2:7b-chat": """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{user_prompt} [/INST] """,
    "mistral:7b-instruct": """<s>[INST]
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{user_prompt} [/INST] """,
    "mixtral:8x7b-instruct": """<s>[INST]
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{user_prompt} [/INST] """,
    "phi3:3.8b-mini-instruct-4k-fp16": """<|user|>
{user_prompt}<|end|>
<|assistant|>""",
}

MODEL_ID_MAP = {
    "llama2:7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "mistral:7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral:8x7b-instruct": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "phi3:3.8b-mini-instruct-4k-fp16": "microsoft/Phi-3-mini-4k-instruct",
}

SERVICE_CONFIG_MAP = {
    "llama2:7b-chat": {
        "name": "llama2",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_type": "nvidia-tesla-t4",
        },
    },
    "mistral:7b-instruct": {
        "name": "mistral",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_type": "nvidia-l4",
        },
    },
    "mixtral:8x7b-instruct": {
        "name": "mixtral",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_type": "nvidia-a100-80gb",
        },
    },
    "phi3:3.8b-mini-instruct-4k-fp16": {
        "name": "phi3",
        "traffic": {
            "timeout": 300,
        },
        "resources": {
            "gpu": 1,
            "gpu_memory": "16GB",
        },
    },
}

MODEL_CONFIG_MAP = {
    "llama2:7b-chat": {
        "max_model_len": 1024,
    },
    "mistral:7b-instruct": {
        "max_model_len": 1024,
        "dtype": "half",
    },
    "mixtral:8x7b-instruct": {
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
        "quantization": "gptq",
        "dtype": "half",
    },
    "phi3:3.8b-mini-instruct-4k-fp16": {
        "max_model_len": 4096,
        "dtype": "half",
    },
}

MODEL_ALIAS_MAP = {
    "llama2": "llama2:7b-chat",
    "llama2:7b": "llama2:7b-chat",
    "llama2:7b-chat": "llama2:7b-chat",
    "mistral": "mistral:7b-instruct",
    "mistral:7b": "mistral:7b-instruct",
    "mistral:7b-instruct": "mistral:7b-instruct",
    "mixtral": "mixtral:8x7b-instruct",
    "mixtral:8x7b": "mixtral:8x7b-instruct",
    "mixtral:8x7b-instruct": "mixtral:8x7b-instruct",
    "phi3": "phi3:3.8b-mini-instruct-4k-fp16",
    "phi3:3.8b": "phi3:3.8b-mini-instruct-4k-fp16",
    "phi3:mini": "phi3:3.8b-mini-instruct-4k-fp16",
    "phi3:instruct": "phi3:3.8b-mini-instruct-4k-fp16",
    "phi3:3.8b-mini-instruct": "phi3:3.8b-mini-instruct-4k-fp16",
}



models:
  "phi3:3.8b-mini-instruct-4k-fp16":
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
    engine_config:
      model_id: microsoft/Phi-3-mini-4k-instruct
      max_model_len: 4096
      dtype: half
