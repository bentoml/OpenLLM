![Banner for OpenLLM](/.github/assets/main-banner.png)

<!-- hatch-fancy-pypi-readme intro start -->

<div align="center">
    <h1 align="center">🦾 OpenLLM</h1>
    <a href="https://pypi.org/project/openllm">
        <img src="https://img.shields.io/pypi/v/openllm.svg?logo=pypi&label=PyPI&logoColor=gold" alt="pypi_status" />
    </a><a href="https://test.pypi.org/project/openllm/">
        <img src="https://img.shields.io/badge/Nightly-PyPI?logo=pypi&label=PyPI&color=gray&link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Fopenllm%2F" alt="test_pypi_status" />
    </a><a href="https://twitter.com/bentomlai">
        <img src="https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow%20Us" alt="Twitter" />
    </a><a href="https://l.bentoml.com/join-openllm-discord">
        <img src="https://badgen.net/badge/icon/OpenLLM/7289da?icon=discord&label=Join%20Us" alt="Discord" />
    </a><a href="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml">
        <img src="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml/badge.svg?branch=main" alt="ci" />
    </a><a href="https://results.pre-commit.ci/latest/github/bentoml/OpenLLM/main">
        <img src="https://results.pre-commit.ci/badge/github/bentoml/OpenLLM/main.svg" alt="pre-commit.ci status" />
    </a><br>
    <a href="https://pypi.org/project/openllm">
        <img src="https://img.shields.io/pypi/pyversions/openllm.svg?logo=python&label=Python&logoColor=gold" alt="python_version" />
    </a><a href="https://github.com/pypa/hatch">
        <img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" alt="Hatch" />
    </a><a href="https://github.com/bentoml/OpenLLM/blob/main/STYLE.md">
        <img src="https://img.shields.io/badge/code%20style-Google-000000.svg" alt="code style" />
    </a><a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Ruff" />
    </a><a href="https://github.com/python/mypy">
        <img src="https://img.shields.io/badge/types-mypy-blue.svg" alt="types - mypy" />
    </a><a href="https://github.com/microsoft/pyright">
        <img src="https://img.shields.io/badge/types-pyright-yellow.svg" alt="types - pyright" />
    </a><br>
    <p>An open platform for operating large language models (LLMs) in production.</br>
    Fine-tune, serve, deploy, and monitor any LLMs with ease.</p>
    <i></i>
</div>

## 📖 Introduction

OpenLLM is an open-source platform designed to facilitate the deployment and operation of large language models (LLMs) in real-world applications. With OpenLLM, you can run inference on any open-source LLM, deploy them on the cloud or on-premises, and build powerful AI applications.

Key features include:

🚂 **State-of-the-art LLMs**: Integrated support for a wide range of open-source LLMs and model runtimes, including but not limited to Llama 2, StableLM, Falcon, Dolly, Flan-T5, ChatGLM, and StarCoder.

🔥 **Flexible APIs**: Serve LLMs over a RESTful API or gRPC with a single command. You can interact with the model using a Web UI, CLI, Python/JavaScript clients, or any HTTP client of your choice.

⛓️ **Freedom to build**: First-class support for LangChain, BentoML and Hugging Face, allowing you to easily create your own AI applications by composing LLMs with other models and services.

🎯 **Streamline deployment**: Automatically generate your LLM server Docker images or deploy as serverless endpoints via
[☁️ BentoCloud](https://l.bentoml.com/bento-cloud), which effortlessly manages GPU resources, scales according to traffic, and ensures cost-effectiveness.

🤖️ **Bring your own LLM**: Fine-tune any LLM to suit your needs. You can load LoRA layers to fine-tune models for higher accuracy and performance for specific tasks. A unified fine-tuning API for models (`LLM.tuning()`) is coming soon.

⚡ **Quantization**: Run inference with less computational and memory costs though quantization techniques like [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPTQ](https://arxiv.org/abs/2210.17323).

📡 **Streaming**: Support token streaming through server-sent events (SSE). You can use the `/v1/generate_stream` endpoint for streaming responses from LLMs.

🔄 **Continuous batching**: Support continuous batching via [vLLM](https://github.com/vllm-project/vllm) for increased total throughput.

OpenLLM is designed for AI application developers working to build production-ready applications based on LLMs. It delivers a comprehensive suite of tools and features for fine-tuning, serving, deploying, and monitoring these models, simplifying the end-to-end deployment workflow for LLMs.

<!-- hatch-fancy-pypi-readme intro stop -->

![Gif showing OpenLLM Intro](/.github/assets/output.gif)

<br/>

<!-- hatch-fancy-pypi-readme interim start -->

## 🏃 Get started

### Prerequisites

You have installed Python 3.8 (or later) and `pip`. We highly recommend using a [Virtual Environment](https://docs.python.org/3/library/venv.html) to prevent package conflicts.

### Install OpenLLM

Install OpenLLM by using `pip` as follows:

```bash
pip install openllm
```

To verify the installation, run:

```bash
$ openllm -h

Usage: openllm [OPTIONS] COMMAND [ARGS]...

   ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
  ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝.

  An open platform for operating large language models in production.
  Fine-tune, serve, deploy, and monitor any LLMs with ease.

Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  build       Package a given models into a Bento.
  embed       Get embeddings interactively, from a terminal.
  import      Setup LLM interactively.
  instruct    Instruct agents interactively for given tasks, from a...
  models      List all supported models.
  prune       Remove all saved models, (and optionally bentos) built with...
  query       Ask a LLM interactively, from a terminal.
  start       Start any LLM as a REST server.
  start-grpc  Start any LLM as a gRPC server.

Extensions:
  build-base-container  Base image builder for BentoLLM.
  dive-bentos           Dive into a BentoLLM.
  get-containerfile     Return Containerfile of any given Bento.
  get-prompt            Get the default prompt used by OpenLLM.
  list-bentos           List available bentos built by OpenLLM.
  list-models           This is equivalent to openllm models...
  playground            OpenLLM Playground.
```

### Start an LLM server

OpenLLM allows you to quickly spin up an LLM server using `openllm start`. For example, to start an [OPT](https://huggingface.co/docs/transformers/model_doc/opt) server, run the following:

```bash
openllm start opt
```

This starts the server at [http://0.0.0.0:3000/](http://0.0.0.0:3000/). OpenLLM downloads the model to the BentoML local Model Store if they have not been registered before. To view your local models, run `bentoml models list`.

To interact with the server, you can visit the web UI at [http://0.0.0.0:3000/](http://0.0.0.0:3000/) or send a request using `curl`. You can also use OpenLLM’s built-in Python client to interact with the server:

```python
import openllm
client = openllm.client.HTTPClient('http://localhost:3000')
client.query('Explain to me the difference between "further" and "farther"')
```

Alternatively, use the `openllm query` command to query the model:

```bash
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'Explain to me the difference between "further" and "farther"'
```

OpenLLM seamlessly supports many models and their variants. You can specify different variants of the model to be served by providing the `--model-id` option. For example:

```bash
openllm start opt --model-id facebook/opt-2.7b
```

> [!NOTE]
> OpenLLM supports specifying fine-tuning weights and quantized weights
> for any of the supported models as long as they can be loaded with the model
> architecture. Use the `openllm models` command to see the complete list of supported
> models, their architectures, and their variants.

## 🧩 Supported models

OpenLLM currently supports the following models. By default, OpenLLM doesn't include dependencies to run all models. The extra model-specific dependencies can be installed with the instructions below.

<details>
<summary>Llama</summary>

### Installation

To run Llama models with OpenLLM, you need to install the `llama` dependency as it is not installed by default.

```bash
pip install "openllm[llama]"
```

### Quickstart

Run the following commands to quickly spin up a Llama 2 server and send a request to it.

```bash
openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

> [!NOTE]
> To use the official Llama 2 models, you must gain access by visiting
> the [Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and
> accepting its license terms and acceptable use policy. You also need to obtain access to these
> models on [Hugging Face](https://huggingface.co/meta-llama). Note that any Llama 2 variants can
> be deployed with OpenLLM if you don’t have access to the official Llama 2 model.
> Visit the [Hugging Face Model Hub](https://huggingface.co/models?sort=trending&search=llama2) to see more Llama 2 compatible models.

### Supported models

You can specify any of the following Llama models by using `--model-id`.

- [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [NousResearch/llama-2-70b-chat-hf](https://huggingface.co/NousResearch/llama-2-70b-chat-hf)
- [NousResearch/llama-2-13b-chat-hf](https://huggingface.co/NousResearch/llama-2-13b-chat-hf)
- [NousResearch/llama-2-7b-chat-hf](https://huggingface.co/NousResearch/llama-2-7b-chat-hf)
- [NousResearch/llama-2-70b-hf](https://huggingface.co/NousResearch/llama-2-70b-hf)
- [NousResearch/llama-2-13b-hf](https://huggingface.co/NousResearch/llama-2-13b-hf)
- [NousResearch/llama-2-7b-hf](https://huggingface.co/NousResearch/llama-2-7b-hf)
- [openlm-research/open_llama_7b_v2](https://huggingface.co/openlm-research/open_llama_7b_v2)
- [openlm-research/open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2)
- [openlm-research/open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b)
- [huggyllama/llama-65b](https://huggingface.co/huggyllama/llama-65b)
- [huggyllama/llama-30b](https://huggingface.co/huggyllama/llama-30b)
- [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b)
- [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
- Any other models that strictly follows the [LlamaForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaForCausalLM) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf --backend pt
  ```

- vLLM (Recommended):

  ```bash
  pip install "openllm[llama, vllm]"
  openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>ChatGLM</summary>

### Installation

To run ChatGLM models with OpenLLM, you need to install the `chatglm` dependency as it is not installed by default.

```bash
pip install "openllm[chatglm]"
```

### Quickstart

Run the following commands to quickly spin up a ChatGLM server and send a request to it.

```bash
openllm start chatglm --model-id thudm/chatglm-6b
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following ChatGLM models by using `--model-id`.

- [thudm/chatglm-6b](https://huggingface.co/thudm/chatglm-6b)
- [thudm/chatglm-6b-int8](https://huggingface.co/thudm/chatglm-6b-int8)
- [thudm/chatglm-6b-int4](https://huggingface.co/thudm/chatglm-6b-int4)
- [thudm/chatglm2-6b](https://huggingface.co/thudm/chatglm2-6b)
- [thudm/chatglm2-6b-int4](https://huggingface.co/thudm/chatglm2-6b-int4)
- Any other models that strictly follows the [ChatGLMForConditionalGeneration](https://github.com/THUDM/ChatGLM-6B) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start chatglm --model-id thudm/chatglm-6b --backend pt
  ```

</details>

<details>
<summary>Dolly-v2</summary>

### Installation

Dolly-v2 models do not require you to install any model-specific dependencies once you have `openllm` installed.

```bash
pip install openllm
```

### Quickstart

Run the following commands to quickly spin up a Dolly-v2 server and send a request to it.

```bash
openllm start dolly-v2 --model-id databricks/dolly-v2-3b
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following Dolly-v2 models by using `--model-id`.

- [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b)
- [databricks/dolly-v2-7b](https://huggingface.co/databricks/dolly-v2-7b)
- [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)
- Any other models that strictly follows the [GPTNeoXForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start dolly-v2 --model-id databricks/dolly-v2-3b --backend pt
  ```

- vLLM:

  ```bash
  openllm start dolly-v2 --model-id databricks/dolly-v2-3b --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>Falcon</summary>

### Installation

To run Falcon models with OpenLLM, you need to install the `falcon` dependency as it is not installed by default.

```bash
pip install "openllm[falcon]"
```

### Quickstart

Run the following commands to quickly spin up a Falcon server and send a request to it.

```bash
openllm start falcon --model-id tiiuae/falcon-7b
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following Falcon models by using `--model-id`.

- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)
- [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
- [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)
- Any other models that strictly follows the [FalconForCausalLM](https://falconllm.tii.ae/) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start falcon --model-id tiiuae/falcon-7b --backend pt
  ```

- vLLM:

  ```bash
  pip install "openllm[falcon, vllm]"
  openllm start falcon --model-id tiiuae/falcon-7b --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>Flan-T5</summary>

### Installation

To run Flan-T5 models with OpenLLM, you need to install the `flan-t5` dependency as it is not installed by default.

```bash
pip install "openllm[flan-t5]"
```

### Quickstart

Run the following commands to quickly spin up a Flan-T5 server and send a request to it.

```bash
openllm start flan-t5 --model-id google/flan-t5-large
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following Flan-T5 models by using `--model-id`.

- [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
- [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
- [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)
- Any other models that strictly follows the [T5ForConditionalGeneration](https://huggingface.co/docs/transformers/main/model_doc/t5#transformers.T5ForConditionalGeneration) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start flan-t5 --model-id google/flan-t5-large --backend pt
  ```

- Flax:

  ```bash
  pip install "openllm[flan-t5, flax]"
  openllm start flan-t5 --model-id google/flan-t5-large --backend flax
  ```

- TensorFlow:

  ```bash
  pip install "openllm[flan-t5, tf]"
  openllm start flan-t5 --model-id google/flan-t5-large --backend tf
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>GPT-NeoX</summary>

### Installation

GPT-NeoX models do not require you to install any model-specific dependencies once you have `openllm` installed.

```bash
pip install openllm
```

### Quickstart

Run the following commands to quickly spin up a GPT-NeoX server and send a request to it.

```bash
openllm start gpt-neox --model-id eleutherai/gpt-neox-20b
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following GPT-NeoX models by using `--model-id`.

- [eleutherai/gpt-neox-20b](https://huggingface.co/eleutherai/gpt-neox-20b)
- Any other models that strictly follows the [GPTNeoXForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start gpt-neox --model-id eleutherai/gpt-neox-20b --backend pt
  ```

- vLLM:

  ```bash
  openllm start gpt-neox --model-id eleutherai/gpt-neox-20b --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>MPT</summary>

### Installation

To run MPT models with OpenLLM, you need to install the `mpt` dependency as it is not installed by default.

```bash
pip install "openllm[mpt]"
```

### Quickstart

Run the following commands to quickly spin up a MPT server and send a request to it.

```bash
openllm start mpt --model-id mosaicml/mpt-7b-chat
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following MPT models by using `--model-id`.

- [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)
- [mosaicml/mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)
- [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
- [mosaicml/mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- [mosaicml/mpt-30b](https://huggingface.co/mosaicml/mpt-30b)
- [mosaicml/mpt-30b-instruct](https://huggingface.co/mosaicml/mpt-30b-instruct)
- [mosaicml/mpt-30b-chat](https://huggingface.co/mosaicml/mpt-30b-chat)
- Any other models that strictly follows the [MPTForCausalLM](https://huggingface.co/mosaicml) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start mpt --model-id mosaicml/mpt-7b-chat --backend pt
  ```

- vLLM (Recommended):

  ```bash
  pip install "openllm[mpt, vllm]"
  openllm start mpt --model-id mosaicml/mpt-7b-chat --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>OPT</summary>

### Installation

To run OPT models with OpenLLM, you need to install the `opt` dependency as it is not installed by default.

```bash
pip install "openllm[opt]"
```

### Quickstart

Run the following commands to quickly spin up an OPT server and send a request to it.

```bash
openllm start opt --model-id facebook/opt-2.7b
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following OPT models by using `--model-id`.

- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-350m](https://huggingface.co/facebook/opt-350m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)
- Any other models that strictly follows the [OPTForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/opt#transformers.OPTForCausalLM) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start opt --model-id facebook/opt-2.7b --backend pt
  ```

- vLLM:

  ```bash
  pip install "openllm[opt, vllm]"
  openllm start opt --model-id facebook/opt-2.7b --backend vllm
  ```

- TensorFlow:

  ```bash
  pip install "openllm[opt, tf]"
  openllm start opt --model-id facebook/opt-2.7b --backend tf
  ```

- Flax:

  ```bash
  pip install "openllm[opt, flax]"
  openllm start opt --model-id facebook/opt-2.7b --backend flax
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>StableLM</summary>

### Installation

StableLM models do not require you to install any model-specific dependencies once you have `openllm` installed.

```bash
pip install openllm
```

### Quickstart

Run the following commands to quickly spin up a StableLM server and send a request to it.

```bash
openllm start stablelm --model-id stabilityai/stablelm-tuned-alpha-7b
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following StableLM models by using `--model-id`.

- [stabilityai/stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
- [stabilityai/stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
- [stabilityai/stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)
- [stabilityai/stablelm-base-alpha-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)
- Any other models that strictly follows the [GPTNeoXForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start stablelm --model-id stabilityai/stablelm-tuned-alpha-7b --backend pt
  ```

- vLLM:

  ```bash
  openllm start stablelm --model-id stabilityai/stablelm-tuned-alpha-7b --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>StarCoder</summary>

### Installation

To run StarCoder models with OpenLLM, you need to install the `starcoder` dependency as it is not installed by default.

```bash
pip install "openllm[starcoder]"
```

### Quickstart

Run the following commands to quickly spin up a StarCoder server and send a request to it.

```bash
openllm start startcoder --model-id [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following StarCoder models by using `--model-id`.

- [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
- [bigcode/starcoderbase](https://huggingface.co/bigcode/starcoderbase)
- Any other models that strictly follows the [GPTBigCodeForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/gpt_bigcode#transformers.GPTBigCodeForCausalLM) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start startcoder --model-id bigcode/starcoder --backend pt
  ```

- vLLM:

  ```bash
  pip install "openllm[startcoder, vllm]"
  openllm start startcoder --model-id bigcode/starcoder --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

<details>
<summary>Baichuan</summary>

### Installation

To run Baichuan models with OpenLLM, you need to install the `baichuan` dependency as it is not installed by default.

```bash
pip install "openllm[baichuan]"
```

### Quickstart

Run the following commands to quickly spin up a Baichuan server and send a request to it.

```bash
openllm start baichuan --model-id baichuan-inc/baichuan-13b-base
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```

### Supported models

You can specify any of the following Baichuan models by using `--model-id`.

- [baichuan-inc/baichuan-7b](https://huggingface.co/baichuan-inc/baichuan-7b)
- [baichuan-inc/baichuan-13b-base](https://huggingface.co/baichuan-inc/baichuan-13b-base)
- [baichuan-inc/baichuan-13b-chat](https://huggingface.co/baichuan-inc/baichuan-13b-chat)
- [fireballoon/baichuan-vicuna-chinese-7b](https://huggingface.co/fireballoon/baichuan-vicuna-chinese-7b)
- [fireballoon/baichuan-vicuna-7b](https://huggingface.co/fireballoon/baichuan-vicuna-7b)
- [hiyouga/baichuan-7b-sft](https://huggingface.co/hiyouga/baichuan-7b-sft)
- Any other models that strictly follows the [BaiChuanForCausalLM](https://github.com/baichuan-inc/Baichuan-7B) architecture

### Supported backends

- PyTorch (Default):

  ```bash
  openllm start baichuan --model-id baichuan-inc/baichuan-13b-base --backend pt
  ```

- vLLM:

  ```bash
  pip install "openllm[baichuan, vllm]"
  openllm start baichuan --model-id baichuan-inc/baichuan-13b-base --backend vllm
  ```

> [!NOTE]
> Currently when using the vLLM backend, quantization and adapters are not supported.

</details>

More models will be integrated with OpenLLM and we welcome your contributions if you want to incorporate your custom LLMs into the ecosystem. Check out [Adding a New Model Guide](https://github.com/bentoml/OpenLLM/blob/main/openllm-python/ADDING_NEW_MODEL.md) to learn more.

## 💻 Run your model on multiple GPUs

OpenLLM allows you to start your model server on multiple GPUs and specify the number of workers per resource assigned using the `--workers-per-resource` option. For example, if you have 4 available GPUs, you set the value as one divided by the number as only one instance of the Runner server will be spawned.

```bash
openllm start opt --workers-per-resource 0.25
```

> [!NOTE]
> The amount of GPUs required depends on the model size itself.
> You can use [the Model Memory Calculator from Hugging Face](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) to
> calculate how much vRAM is needed to train and perform big model
> inference on a model and then plan your GPU strategy based on it.

When using the `--workers-per-resource` option with the `openllm build` command, the environment variable is saved into the resulting Bento.

For more information, see [Resource scheduling strategy](https://docs.bentoml.org/en/latest/guides/scheduling.html#).

## 🛞 Runtime implementations (Experimental)

Different LLMs may support multiple runtime implementations. For instance, they might use frameworks and libraries such as PyTorch (`pt`), TensorFlow (`tf`), Flax (`flax`), and vLLM (`vllm`).

To specify a specific runtime for your chosen model, use the `--backend` option. For example:

```bash
openllm start llama --model-id meta-llama/Llama-2-7b-chat-hf --backend vllm
```

Note:

1. For GPU support on Flax, refers to [Jax's installation](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) to make sure that you have Jax support for the corresponding CUDA version.
2. To use the vLLM backend, you need a GPU with at least the Ampere architecture or newer and CUDA version 11.8.
3. To see the backend options of each model supported by OpenLLM, see the Supported models section or run `openllm models`.

## 📐 Quantization

Quantization is a technique to reduce the storage and computation requirements for machine learning models, particularly during inference. By approximating floating-point numbers as integers (quantized values), quantization allows for faster computations, reduced memory footprint, and can make it feasible to deploy large models on resource-constrained devices.

OpenLLM supports quantization through two methods - [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [GPTQ](https://arxiv.org/abs/2210.17323).

To run a model using the `bitsandbytes` method for quantization, you can use the following command:

```bash
openllm start opt --quantize int8
```

To run inference with `gptq`, simply pass `--quantize gptq`:

```bash
openllm start falcon --model-id TheBloke/falcon-40b-instruct-GPTQ --quantize gptq --device 0
```

> [!NOTE]
> In order to run GPTQ, make sure you run `pip install "openllm[gptq]" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`
> first to install the dependency. From the GPTQ paper, it is recommended to quantized the weights before serving.
> See [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) for more information on GPTQ quantization.

## 🛠️ Fine-tuning support (Experimental)

[PEFT](https://huggingface.co/docs/peft/index), or Parameter-Efficient Fine-Tuning, is a methodology designed to fine-tune pre-trained models more efficiently. Instead of adjusting all model parameters, PEFT focuses on tuning only a subset, reducing computational and storage costs. [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora) (Low-Rank Adaptation) is one of the techniques supported by PEFT. It streamlines fine-tuning by using low-rank decomposition to represent weight updates, thereby drastically reducing the number of trainable parameters.

With OpenLLM, you can take advantage of the fine-tuning feature by serving models with any PEFT-compatible layers using the `--adapter-id` option. For example:

```bash
openllm start opt --model-id facebook/opt-6.7b --adapter-id aarnphm/opt-6-7b-quotes
```

OpenLLM also provides flexibility by supporting adapters from custom file paths:

```bash
openllm start opt --model-id facebook/opt-6.7b --adapter-id /path/to/adapters
```

To use multiple adapters, use the following format:

```bash
openllm start opt --model-id facebook/opt-6.7b --adapter-id aarnphm/opt-6.7b-lora --adapter-id aarnphm/opt-6.7b-lora:french_lora
```

By default, the first specified `adapter-id` is the default LoRA layer, but optionally you can specify a different LoRA layer for inference using the `/v1/adapters` endpoint:

```bash
curl -X POST http://localhost:3000/v1/adapters --json '{"adapter_name": "vn_lora"}'
```

Note that if you are using multiple adapter names and IDs, it is recommended to set the default adapter before sending the inference to avoid any performance degradation.

To include this into the Bento, you can specify the `--adapter-id` option when using the `openllm build` command:

```bash
openllm build opt --model-id facebook/opt-6.7b --adapter-id ...
```

If you use a relative path for `--adapter-id`, you need to add `--build-ctx`.

```bash
openllm build opt --adapter-id ./path/to/adapter_id --build-ctx .
```

> [!NOTE]
> We will gradually roll out support for fine-tuning all models.
> Currently, the models supporting fine-tuning with OpenLLM include: OPT, Falcon, and LlaMA.

## 🧮 Embeddings

OpenLLM provides embeddings endpoint for embeddings calculation. This can
be accessed via `/v1/embeddings`.

To use via CLI, simply call `openllm embed`:

```bash
openllm embed --endpoint http://localhost:3000 "I like to eat apples" -o json
{
  "embeddings": [
    0.006569798570126295,
    -0.031249752268195152,
    -0.008072729222476482,
    0.00847396720200777,
    -0.005293501541018486,
    ...<many embeddings>...
    -0.002078012563288212,
    -0.00676426338031888,
    -0.002022686880081892
  ],
  "num_tokens": 9
}
```

To invoke this endpoint, use `client.embed` from the Python SDK:

```python
import openllm

client = openllm.client.HTTPClient("http://localhost:3000")

client.embed("I like to eat apples")
```

> [!NOTE]
> Currently, the following model family supports embeddings calculation: Llama, T5 (Flan-T5, FastChat, etc.), ChatGLM
> For the remaining LLM that doesn't have specific embedding implementation,
> we will use a generic [BertModel](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
> for embeddings generation. The implementation is largely based on [`bentoml/sentence-embedding-bento`](https://github.com/bentoml/sentence-embedding-bento)

## 🥅 Playground and Chat UI

The following UIs are currently available for OpenLLM:

| UI                                                                                        | Owner                                        | Type                 | Progress |
| ----------------------------------------------------------------------------------------- | -------------------------------------------- | -------------------- | -------- |
| [Clojure](https://github.com/bentoml/OpenLLM/blob/main/openllm-contrib/clojure/README.md) | [@GutZuFusss](https://github.com/GutZuFusss) | Community-maintained | 🔧       |
| TS                                                                                        | BentoML Team                                 |                      | 🚧       |

## ⚙️ Integrations

OpenLLM is not just a standalone product; it's a building block designed to
integrate with other powerful tools easily. We currently offer integration with
[BentoML](https://github.com/bentoml/BentoML),
[LangChain](https://github.com/hwchase17/langchain), and
[Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents).

### BentoML

OpenLLM models can be integrated as a
[Runner](https://docs.bentoml.com/en/latest/concepts/runner.html) in your
BentoML service. These runners have a `generate` method that takes a string as a
prompt and returns a corresponding output string. This will allow you to plug
and play any OpenLLM models with your existing ML workflow.

```python
import bentoml
import openllm

model = "opt"

llm_config = openllm.AutoConfig.for_model(model)
llm_runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(
    name=f"llm-opt-service", runners=[llm_runner]
)

@svc.api(input=Text(), output=Text())
async def prompt(input_text: str) -> str:
    answer = await llm_runner.generate(input_text)
    return answer
```

### [LangChain](https://python.langchain.com/docs/ecosystem/integrations/openllm)

To quickly start a local LLM with `langchain`, simply do the following:

```python
from langchain.llms import OpenLLM

llm = OpenLLM(model_name="llama", model_id='meta-llama/Llama-2-7b-hf')

llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
```

> [!IMPORTANT]
> By default, OpenLLM use `safetensors` format for saving models.
> If the model doesn't support safetensors, make sure to pass
> `serialisation="legacy"` to use the legacy PyTorch bin format.

`langchain.llms.OpenLLM` has the capability to interact with remote OpenLLM
Server. Given there is an OpenLLM server deployed elsewhere, you can connect to
it by specifying its URL:

```python
from langchain.llms import OpenLLM

llm = OpenLLM(server_url='http://44.23.123.1:3000', server_type='grpc')
llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
```

To integrate a LangChain agent with BentoML, you can do the following:

```python
llm = OpenLLM(
    model_name='flan-t5',
    model_id='google/flan-t5-large',
    embedded=False,
    serialisation="legacy"
)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
svc = bentoml.Service("langchain-openllm", runners=[llm.runner])
@svc.api(input=Text(), output=Text())
def chat(input_text: str):
    return agent.run(input_text)
```

> [!NOTE]
> You can find out more examples under the
> [examples](https://github.com/bentoml/OpenLLM/tree/main/examples) folder.

### Transformers Agents

OpenLLM seamlessly integrates with
[Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents).

> [!WARNING]
> The Transformers Agent is still at an experimental stage. It is
> recommended to install OpenLLM with `pip install -r nightly-requirements.txt`
> to get the latest API update for HuggingFace agent.

```python
import transformers

agent = transformers.HfAgent("http://localhost:3000/hf/agent")  # URL that runs the OpenLLM server

agent.run("Is the following `text` positive or negative?", text="I don't like how this models is generate inputs")
```

> [!IMPORTANT]
> Only `starcoder` is currently supported with Agent integration.
> The example above was also run with four T4s on EC2 `g4dn.12xlarge`

If you want to use OpenLLM client to ask questions to the running agent, you can
also do so:

```python
import openllm

client = openllm.client.HTTPClient("http://localhost:3000")

client.ask_agent(
    task="Is the following `text` positive or negative?",
    text="What are you thinking about?",
)
```

<!-- hatch-fancy-pypi-readme interim stop -->

![Gif showing Agent integration](/.github/assets/agent.gif)

<br/>

<!-- hatch-fancy-pypi-readme meta start -->

## 🚀 Deploying models to production

There are several ways to deploy your LLMs:

### 🐳 Docker container

1. **Building a Bento**: With OpenLLM, you can easily build a Bento for a
   specific model, like `dolly-v2`, using the `build` command.:

   ```bash
   openllm build dolly-v2
   ```

   A
   [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html#what-is-a-bento),
   in BentoML, is the unit of distribution. It packages your program's source
   code, models, files, artefacts, and dependencies.

2. **Containerize your Bento**

   ```bash
   bentoml containerize <name:version>
   ```

   This generates a OCI-compatible docker image that can be deployed anywhere
   docker runs. For best scalability and reliability of your LLM service in
   production, we recommend deploy with BentoCloud。

### ☁️ BentoCloud

Deploy OpenLLM with [BentoCloud](https://www.bentoml.com/bento-cloud/), the
serverless cloud for shipping and scaling AI applications.

1. **Create a BentoCloud account:** [sign up here](https://bentoml.com/cloud)
   for early access

2. **Log into your BentoCloud account:**

   ```bash
   bentoml cloud login --api-token <your-api-token> --endpoint <bento-cloud-endpoint>
   ```

> [!NOTE]
> Replace `<your-api-token>` and `<bento-cloud-endpoint>` with your
> specific API token and the BentoCloud endpoint respectively.

3. **Bulding a Bento**: With OpenLLM, you can easily build a Bento for a
   specific model, such as `dolly-v2`:

   ```bash
   openllm build dolly-v2
   ```

4. **Pushing a Bento**: Push your freshly-built Bento service to BentoCloud via
   the `push` command:

   ```bash
   bentoml push <name:version>
   ```

5. **Deploying a Bento**: Deploy your LLMs to BentoCloud with a single
   `bentoml deployment create` command following the
   [deployment instructions](https://docs.bentoml.com/en/latest/reference/cli.html#bentoml-deployment-create).

## 👥 Community

Engage with like-minded individuals passionate about LLMs, AI, and more on our
[Discord](https://l.bentoml.com/join-openllm-discord)!

OpenLLM is actively maintained by the BentoML team. Feel free to reach out and
join us in our pursuit to make LLMs more accessible and easy to use 👉
[Join our Slack community!](https://l.bentoml.com/join-slack)

## 🎁 Contributing

We welcome contributions! If you're interested in enhancing OpenLLM's
capabilities or have any questions, don't hesitate to reach out in our
[discord channel](https://l.bentoml.com/join-openllm-discord).

Checkout our
[Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md)
if you wish to contribute to OpenLLM's codebase.

## 🍇 Telemetry

OpenLLM collects usage data to enhance user experience and improve the product.
We only report OpenLLM's internal API calls and ensure maximum privacy by
excluding sensitive information. We will never collect user code, model data, or
stack traces. For usage tracking, check out the
[code](https://github.com/bentoml/OpenLLM/blob/main/openllm-python/src/openllm/utils/analytics.py).

You can opt out of usage tracking by using the `--do-not-track` CLI option:

```bash
openllm [command] --do-not-track
```

Or by setting the environment variable `OPENLLM_DO_NOT_TRACK=True`:

```bash
export OPENLLM_DO_NOT_TRACK=True
```

## 📔 Citation

If you use OpenLLM in your research, we provide a [citation](./CITATION.cff) to
use:

```bibtex
@software{Pham_OpenLLM_Operating_LLMs_2023,
author = {Pham, Aaron and Yang, Chaoyu and Sheng, Sean and  Zhao, Shenyang and Lee, Sauyon and Jiang, Bo and Dong, Fog and Guan, Xipeng and Ming, Frost},
license = {Apache-2.0},
month = jun,
title = {{OpenLLM: Operating LLMs in production}},
url = {https://github.com/bentoml/OpenLLM},
year = {2023}
}
```

<!-- hatch-fancy-pypi-readme meta stop -->
