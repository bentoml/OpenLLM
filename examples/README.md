## Examples with OpenLLM

You can find the following examples to interact with OpenLLM features. See more [here](../README.md)

### OpenAI-compatible endpoints

The [`openai_completion_client.py`](./openai_completion_client.py) demos how to use the OpenAI-compatible `/v1/completions` to generate text.

```bash
export OPENLLM_ENDPOINT=https://api.openllm.com
python openai_completion_client.py

# For streaming set STREAM=True
STREAM=True python openai_completion_client.py
```

The [`openai_chat_completion_client.py`](./openai_chat_completion_client.py) demos how to use the OpenAI-compatible `/v1/chat/completions` to chat with a model.

```bash
export OPENLLM_ENDPOINT=https://api.openllm.com
python openai_chat_completion_client.py

# For streaming set STREAM=True
STREAM=True python openai_chat_completion_client.py
```

### TinyLLM

The [`api_server.py`](./api_server.py) demos how one can easily write production-ready BentoML service with OpenLLM and vLLM.

Install requirements:

```bash
pip install -U "openllm[vllm]"
```

To serve the Bento (given you have access to GPU):

```bash
bentoml serve api_server:svc
```

To build the Bento do the following:

```bash
bentoml build -f bentofile.yaml .
```
