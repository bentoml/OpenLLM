<div align="center">
    <h1 align="center">Output structured data with Outlines and BentoML</h1>
</div>

[Outlines](https://github.com/outlines-dev/outlines) is an open-source Python package for structured text generation, integrating with various models to produce controlled, format-specific outputs​. It offers capabilities like fast regex-structured generation, JSON generation following a JSON schema or a Pydantic model, and grammar-structured generation. 

This is a BentoML example project, demonstrating how to output structured data from an LLM using Outlines and BentoML. See [here](https://github.com/bentoml/BentoML?tab=readme-ov-file#%EF%B8%8F-what-you-can-build-with-bentoml) for a full list of BentoML example projects.

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- If you want to test the Service locally, you need a Nvidia GPU with at least 16G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/outlines-integration
pip install -r requirements.txt && pip install -f -U "pydantic>=2.0"
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-03-27T10:14:50+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLM" listening on http://localhost:3000 (Press CTRL+C to quit)
INFO 03-27 10:14:54 llm_engine.py:87] Initializing an LLM engine with config: model='mistralai/Mistral-7B-Instruct-v0.2', tokenizer='mistralai/Mistral-7B-Instruct-v0.2', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=1024, download_dir=None, load_format=auto, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, seed=0)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Create a user profile with the fields name, last_name and id. name should be common English first names. last_name should be common English last names. id should be a random integer",
  "max_tokens": 1024,
  "json_schema": "\n{\n  \"title\": \"User\",\n  \"type\": \"object\",\n  \"properties\": {\n    \"name\": {\"type\": \"string\"},\n    \"last_name\": {\"type\": \"string\"},\n    \"id\": {\"type\": \"integer\"}\n  }\n}\n",
  "regex_string": null
}'
```

Python client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.generate(
        json_schema="\n{\n  \"title\": \"User\",\n  \"type\": \"object\",\n  \"properties\": {\n    \"name\": {\"type\": \"string\"},\n    \"last_name\": {\"type\": \"string\"},\n    \"id\": {\"type\": \"integer\"}\n  }\n}\n",
        max_tokens=1024,
        prompt="Create a user profile with the fields name, last_name and id. name should be common English first names. last_name should be common English last names. id should be a random integer",
        regex_string="",
    )
```

Example output:

```bash
{
 "name": "Oliver",
 "last_name": "Johnson",
 "id": 123456
}
```

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
