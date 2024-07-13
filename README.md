<div align="center">
    <h1 align="center">🦾 OpenLLM: Self-Hosting LLMs Made Easy</h1>
</div>

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202-green.svg)](https://github.com/bentoml/OpenLLM/blob/main/LICENSE)
[![Releases](https://img.shields.io/pypi/v/openllm.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/openllm)
[![CI](https://results.pre-commit.ci/badge/github/bentoml/OpenLLM/main.svg)](https://results.pre-commit.ci/latest/github/bentoml/OpenLLM/main)
[![X](https://badgen.net/badge/icon/@bentomlai/000000?icon=twitter&label=Follow)](https://twitter.com/bentomlai)
[![Community](https://badgen.net/badge/icon/Community/562f5d?icon=slack&label=Join)](https://l.bentoml.com/join-slack)

## Introduction

OpenLLM lets developers run any **open-source LLMs** as **OpenAI-compatible API** endpoints with **a single command**.

- 🔬 Build for fast and production usages
- 🚂 Support llama3, qwen2, gemma, etc, and many quantized versions [full list](https://github.com/bentoml/openllm-models)
- ⛓️ OpenAI-compatible API
- 💬 Built-in ChatGPT like UI
- 🔥 Accelerated LLM decoding with state-of-the-art inference backends
- 🌥️ Ready for enterprise-grade cloud deployment (Kubernetes, Docker and BentoCloud)

## Get started

Install (Python 3.9 and above)

```bash
pip install openllm
```

then run

```bash
openllm hello
```

to explore models interactively. It will guide you to run LLMs either locally or in the cloud.

https://github.com/bentoml/OpenLLM/assets/5886138/2aa5f9e4-859c-4be8-91b3-91d8a810ef73

## Supported models

OpenLLM supports a variety of state-of-the-art LLMs. Here are some of the models supported by OpenLLM, each listed with a commonly used model size.

| Model   | Parameters | Quantinize | Required GPU  | Start a Server                  |
| ------- | ---------- | ---------- | ------------- | ------------------------------- |
| Llama 3 | 8B         | -          | 24G           | `openllm serve llama3:8b`       |
| Llama 3 | 8B         | AWQ 4bit   | 12G           | `openllm serve llama3:8b-4bit`  |
| Llama 3 | 70B        | AWQ 4bit   | 80G           | `openllm serve llama3:70b-4bit` |
| Llama 2 | 7B         | -          | 16G           | `openllm serve llama2:7b`       |
| Llama 2 | 7B         | AWQ 4bit   | 12G           | `openllm serve llama2:7b-4bit`  |
| Mistral | 7B         | -          | 24G           | `openllm serve mistral:7b`      |
| Qwen2   | 1.5B       | -          | 12G           | `openllm serve qwen2:1.5b`      |
| Gemma   | 7B         | -          | 24G           | `openllm serve gemma:7b`        |
| Phi3    | 3.8B       | -          | 12G           | `openllm serve phi3:3.8b`       |

...

For the full model list, see the [OpenLLM models repository](https://github.com/bentoml/openllm-models).

## Start an LLM server

To start an LLM server locally, use the `openllm serve` command and specify the model version.

```bash
openllm serve llama3:8b
```

The server will be accessible at [http://localhost:3000](http://localhost:3000/), providing OpenAI-compatible APIs for interaction. You can call the endpoints with different frameworks and tools that support OpenAI-compatible APIs. Typically, you may need to specify the following:

- **The API host address**: By default, the LLM is hosted at [http://localhost:3000](http://localhost:3000/).
- **The model name:** The name can be different depending on the tool you use.
- **The API key**: The API key used for client authentication. This is optional.

Here are some examples:

<details>

<summary>OpenAI Python client</summary>

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
# model_list = client.models.list()
# print(model_list)

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain superconductors like I'm five years old"
        }
    ],
    stream=True,
)
for chunk in chat_completion:
    print(chunk.choices[0].delta.content or "", end="")
```

</details>


<details>

<summary>LlamaIndex</summary>

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(api_bese="http://localhost:3000/v1", model="meta-llama/Meta-Llama-3-8B-Instruct", api_key="dummy")
...
```
</details>

## Chat UI

OpenLLM provides a chat user interface (UI) at the `/chat` endpoint for an LLM server. You can visit the chat UI at http://localhost:3000/chat and start different conversations with the model.

<img width="800" alt="openllm_ui" src="https://github.com/bentoml/OpenLLM/assets/5886138/8b426b2b-67da-4545-8b09-2dc96ff8a707">

## Chat with a model in the CLI

To start a chat conversation in the CLI, use the `openllm run` command and specify the model version.

```bash
openllm run llama3:8b
```

## Model repository

A model repository in OpenLLM represents a catalog of available LLMs that you can run. OpenLLM provides a default model repository that includes the latest open-source LLMs like Llama 3, Mistral, and Qwen2, hosted at [this GitHub repository](https://github.com/bentoml/openllm-models). To see all available models from the default and any added repository, use:

```bash
openllm model list
```

To ensure your local list of models is synchronized with the latest updates from all connected repositories, run:

```bash
openllm repo update
```

To review a model’s information, run:

```bash
openllm model get llama3:8b
```

### Add a model to the default model repository

You can contribute to the default model repository by adding new models that others can use. This involves creating and submitting a Bento of the LLM. For more information, check out this [example pull request](https://github.com/bentoml/openllm-models/pull/1).

### Set up a custom repository

You can add your own repository to OpenLLM with custom models. To do so, follow the format in the default OpenLLM model repository with a `bentos` directory to store custom LLMs. You need to [build your Bentos with BentoML](https://docs.bentoml.com/en/latest/guides/build-options.html) and submit them to your model repository.

First, prepare your custom models in a `bentos` directory following the guidelines provided by [BentoML to build Bentos](https://docs.bentoml.com/en/latest/guides/build-options.html). Check out the [default model repository](https://github.com/bentoml/openllm-repo) for an example and read the [Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md) for details.

Then, register your custom model repository with OpenLLM:

```bash
openllm repo add <repo-name> <repo-url>
```

**Note**: Currently, OpenLLM only supports adding public repositories.

## Deploy to BentoCloud

OpenLLM supports LLM cloud deployment via BentoML, the unified model serving framework, and BentoCloud, an AI inference platform for enterprise AI teams. BentoCloud provides fully-managed infrastructure optimized for LLM inference with autoscaling, model orchestration, observability, and many more, allowing you to run any AI model in the cloud.

[Sign up for BentoCloud](https://www.bentoml.com/) for free and [log in](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html). Then, run `openllm deploy` to deploy a model to BentoCloud:

```bash
openllm deploy llama3:8b
```

Once the deployment is complete, you can run model inference on the BentoCloud console:

<img width="800" alt="bentocloud_ui" src="https://github.com/bentoml/OpenLLM/assets/65327072/4f7819d9-73ea-488a-a66c-f724e5d063e6">

## Community

OpenLLM is actively maintained by the BentoML team. Feel free to reach out and join us in our pursuit to make LLMs more accessible and easy to use 👉 [Join our Slack community!](https://l.bentoml.com/join-slack)

## Contributing

As an open-source project, we welcome contributions of all kinds, such as new features, bug fixes, and documentation. Here are some of the ways to contribute:

- Repost a bug by [creating a GitHub issue](https://github.com/bentoml/OpenLLM/issues/new/choose).
- [Submit a pull request](https://github.com/bentoml/OpenLLM/compare) or help review other developers’ [pull requests](https://github.com/bentoml/OpenLLM/pulls).
- Add an LLM to the OpenLLM default model repository so that other users can run your model. See the [pull request template](https://github.com/bentoml/openllm-models/pull/1).
- Check out the [Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md) to learn more.

## Acknowledgements

This project uses the following open-source projects:

- [bentoml/bentoml](https://github.com/bentoml/bentoml) for production level model serving
- [vllm-project/vllm](https://github.com/vllm-project/vllm) for production level LLM backend
- [blrchen/chatgpt-lite](https://github.com/blrchen/chatgpt-lite) for a fancy Web Chat UI
- [chujiezheng/chat_templates](https://github.com/chujiezheng/chat_templates)
- [astral-sh/uv](https://github.com/astral-sh/uv) for blazing fast model requirements installing

We are grateful to the developers and contributors of these projects for their hard work and dedication.
