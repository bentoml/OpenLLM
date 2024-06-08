![Banner for OpenLLM](/.github/assets/main-banner.png)

<!-- hatch-fancy-pypi-readme intro start -->

<div align="center">
    <h1 align="center">🦾 OpenLLM: Self-Hosting LLMs Made Easy</h1>
    <a href="https://pypi.org/project/openllm">
        <img src="https://img.shields.io/pypi/v/openllm.svg?logo=pypi&label=PyPI&logoColor=gold" alt="pypi_status" />
    </a><a href="https://test.pypi.org/project/openllm/">
        <img src="https://img.shields.io/badge/Nightly-PyPI?logo=pypi&label=PyPI&color=gray&link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Fopenllm%2F" alt="test_pypi_status" />
    </a><a href="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml">
        <img src="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml/badge.svg?branch=main" alt="ci" />
    </a><a href="https://results.pre-commit.ci/latest/github/bentoml/OpenLLM/main">
        <img src="https://results.pre-commit.ci/badge/github/bentoml/OpenLLM/main.svg" alt="pre-commit.ci status" />
    </a><br><a href="https://twitter.com/bentomlai">
        <img src="https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow%20Us" alt="Twitter" />
    </a><a href="https://l.bentoml.com/join-openllm-discord">
        <img src="https://badgen.net/badge/icon/OpenLLM/7289da?icon=discord&label=Join%20Us" alt="Discord" />
    </a>
</div>

## 📖 Introduction

OpenLLM helps developers **run any open-source LLMs**, such as Llama 2 and Mistral, as **OpenAI-compatible API endpoints**, locally and in the cloud, optimized for serving throughput and production deployment.

- 🚂 Support a wide range of open-source LLMs including LLMs fine-tuned with your own data
- ⛓️ OpenAI compatible API endpoints for seamless transition from your LLM app to open-source LLMs
- 🔥 State-of-the-art serving and inference performance
- 🎯 Simplified cloud deployment via [BentoML](https://www.bentoml.com)

<!-- hatch-fancy-pypi-readme intro stop -->

![Gif showing OpenLLM Intro](/.github/assets/output.gif)

<br/>

<!-- hatch-fancy-pypi-readme interim start -->

## 💾 TL/DR

For starter, we provide two ways to quickly try out OpenLLM:

### Jupyter Notebooks

Try this [OpenLLM tutorial in Google Colab: Serving Phi 3 with OpenLLM](https://colab.research.google.com/github/bentoml/OpenLLM/blob/main/examples/llama2.ipynb).

## 🏃 Get started

The following provides instructions for how to get started with OpenLLM locally.

### Prerequisites

You have installed Python 3.9 (or later) and `pip`. We highly recommend using a [Virtual Environment](https://docs.python.org/3/library/venv.html) to prevent package conflicts.

### Install OpenLLM

Install OpenLLM by using `pip` as follows:

```bash
pip install openllm
```

To verify the installation, run:

```bash
$ openllm -h
```

### Start a LLM server

OpenLLM allows you to quickly spin up an LLM server using `openllm start`. For example, to start a [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) server, run the following:

```bash
openllm start microsoft/Phi-3-mini-4k-instruct --trust-remote-code
```

To interact with the server, you can visit the web UI at [http://0.0.0.0:3000/](http://0.0.0.0:3000/) or send a request using `curl`. You can also use OpenLLM’s built-in Python client to interact with the server:

```python
import openllm

client = openllm.HTTPClient('http://localhost:3000')
client.generate('Explain to me the difference between "further" and "farther"')
```

OpenLLM seamlessly supports many models and their variants. You can specify different variants of the model to be served. For example:

```bash
openllm start <model_id> --<options>
```

## 🧩 Supported models

OpenLLM currently supports the following models. By default, OpenLLM doesn't include dependencies to run all models. The extra model-specific dependencies can be installed with the instructions below.

<!-- update-readme.py: start -->
<details>

<summary>Baichuan</summary>


### Quickstart

Run the following command to quickly spin up a Baichuan server:

```bash
openllm start baichuan-inc/baichuan-7b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Baichuan variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=baichuan) to see more Baichuan-compatible models.



### Supported models

You can specify any of the following Baichuan models via `openllm start`:


- [baichuan-inc/baichuan2-7b-base](https://huggingface.co/baichuan-inc/baichuan2-7b-base)
- [baichuan-inc/baichuan2-7b-chat](https://huggingface.co/baichuan-inc/baichuan2-7b-chat)
- [baichuan-inc/baichuan2-13b-base](https://huggingface.co/baichuan-inc/baichuan2-13b-base)
- [baichuan-inc/baichuan2-13b-chat](https://huggingface.co/baichuan-inc/baichuan2-13b-chat)

</details>

<details>

<summary>ChatGLM</summary>


### Quickstart

Run the following command to quickly spin up a ChatGLM server:

```bash
openllm start thudm/chatglm-6b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any ChatGLM variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=chatglm) to see more ChatGLM-compatible models.



### Supported models

You can specify any of the following ChatGLM models via `openllm start`:


- [thudm/chatglm-6b](https://huggingface.co/thudm/chatglm-6b)
- [thudm/chatglm-6b-int8](https://huggingface.co/thudm/chatglm-6b-int8)
- [thudm/chatglm-6b-int4](https://huggingface.co/thudm/chatglm-6b-int4)
- [thudm/chatglm2-6b](https://huggingface.co/thudm/chatglm2-6b)
- [thudm/chatglm2-6b-int4](https://huggingface.co/thudm/chatglm2-6b-int4)
- [thudm/chatglm3-6b](https://huggingface.co/thudm/chatglm3-6b)

</details>

<details>

<summary>Cohere</summary>


### Quickstart

Run the following command to quickly spin up a Cohere server:

```bash
openllm start CohereForAI/c4ai-command-r-plus --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Cohere variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=commandr) to see more Cohere-compatible models.



### Supported models

You can specify any of the following Cohere models via `openllm start`:


- [CohereForAI/c4ai-command-r-plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
- [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)

</details>

<details>

<summary>Dbrx</summary>


### Quickstart

Run the following command to quickly spin up a Dbrx server:

```bash
openllm start databricks/dbrx-instruct --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Dbrx variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=dbrx) to see more Dbrx-compatible models.



### Supported models

You can specify any of the following Dbrx models via `openllm start`:


- [databricks/dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct)
- [databricks/dbrx-base](https://huggingface.co/databricks/dbrx-base)

</details>

<details>

<summary>DollyV2</summary>


### Quickstart

Run the following command to quickly spin up a DollyV2 server:

```bash
openllm start databricks/dolly-v2-3b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any DollyV2 variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=dolly_v2) to see more DollyV2-compatible models.



### Supported models

You can specify any of the following DollyV2 models via `openllm start`:


- [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b)
- [databricks/dolly-v2-7b](https://huggingface.co/databricks/dolly-v2-7b)
- [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)

</details>

<details>

<summary>Falcon</summary>


### Quickstart

Run the following command to quickly spin up a Falcon server:

```bash
openllm start tiiuae/falcon-7b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Falcon variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=falcon) to see more Falcon-compatible models.



### Supported models

You can specify any of the following Falcon models via `openllm start`:


- [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)
- [tiiuae/falcon-40b](https://huggingface.co/tiiuae/falcon-40b)
- [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
- [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)

</details>

<details>

<summary>Gemma</summary>


### Quickstart

Run the following command to quickly spin up a Gemma server:

```bash
openllm start google/gemma-7b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Gemma variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=gemma) to see more Gemma-compatible models.



### Supported models

You can specify any of the following Gemma models via `openllm start`:


- [google/gemma-7b](https://huggingface.co/google/gemma-7b)
- [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it)
- [google/gemma-2b](https://huggingface.co/google/gemma-2b)
- [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

</details>

<details>

<summary>GPTNeoX</summary>


### Quickstart

Run the following command to quickly spin up a GPTNeoX server:

```bash
openllm start eleutherai/gpt-neox-20b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any GPTNeoX variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=gpt_neox) to see more GPTNeoX-compatible models.



### Supported models

You can specify any of the following GPTNeoX models via `openllm start`:


- [eleutherai/gpt-neox-20b](https://huggingface.co/eleutherai/gpt-neox-20b)

</details>

<details>

<summary>Llama</summary>


### Quickstart

Run the following command to quickly spin up a Llama server:

```bash
openllm start NousResearch/llama-2-7b-hf --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Llama variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=llama) to see more Llama-compatible models.



### Supported models

You can specify any of the following Llama models via `openllm start`:


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

</details>

<details>

<summary>Mistral</summary>


### Quickstart

Run the following command to quickly spin up a Mistral server:

```bash
openllm start mistralai/Mistral-7B-Instruct-v0.1 --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Mistral variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=mistral) to see more Mistral-compatible models.



### Supported models

You can specify any of the following Mistral models via `openllm start`:


- [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)
- [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

</details>

<details>

<summary>Mixtral</summary>


### Quickstart

Run the following command to quickly spin up a Mixtral server:

```bash
openllm start mistralai/Mixtral-8x7B-Instruct-v0.1 --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Mixtral variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=mixtral) to see more Mixtral-compatible models.



### Supported models

You can specify any of the following Mixtral models via `openllm start`:


- [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

</details>

<details>

<summary>MPT</summary>


### Quickstart

Run the following command to quickly spin up a MPT server:

```bash
openllm start mosaicml/mpt-7b-instruct --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any MPT variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=mpt) to see more MPT-compatible models.



### Supported models

You can specify any of the following MPT models via `openllm start`:


- [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)
- [mosaicml/mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)
- [mosaicml/mpt-7b-chat](https://huggingface.co/mosaicml/mpt-7b-chat)
- [mosaicml/mpt-7b-storywriter](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- [mosaicml/mpt-30b](https://huggingface.co/mosaicml/mpt-30b)
- [mosaicml/mpt-30b-instruct](https://huggingface.co/mosaicml/mpt-30b-instruct)
- [mosaicml/mpt-30b-chat](https://huggingface.co/mosaicml/mpt-30b-chat)

</details>

<details>

<summary>OPT</summary>


### Quickstart

Run the following command to quickly spin up a OPT server:

```bash
openllm start facebook/opt-1.3b
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any OPT variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=opt) to see more OPT-compatible models.



### Supported models

You can specify any of the following OPT models via `openllm start`:


- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-350m](https://huggingface.co/facebook/opt-350m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)

</details>

<details>

<summary>Phi</summary>


### Quickstart

Run the following command to quickly spin up a Phi server:

```bash
openllm start microsoft/Phi-3-mini-4k-instruct --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Phi variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=phi) to see more Phi-compatible models.



### Supported models

You can specify any of the following Phi models via `openllm start`:


- [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
- [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct)
- [microsoft/Phi-3-small-128k-instruct](https://huggingface.co/microsoft/Phi-3-small-128k-instruct)
- [microsoft/Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct)
- [microsoft/Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct)

</details>

<details>

<summary>Qwen</summary>


### Quickstart

Run the following command to quickly spin up a Qwen server:

```bash
openllm start qwen/Qwen-7B-Chat --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Qwen variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=qwen) to see more Qwen-compatible models.



### Supported models

You can specify any of the following Qwen models via `openllm start`:


- [qwen/Qwen-7B-Chat](https://huggingface.co/qwen/Qwen-7B-Chat)
- [qwen/Qwen-7B-Chat-Int8](https://huggingface.co/qwen/Qwen-7B-Chat-Int8)
- [qwen/Qwen-7B-Chat-Int4](https://huggingface.co/qwen/Qwen-7B-Chat-Int4)
- [qwen/Qwen-14B-Chat](https://huggingface.co/qwen/Qwen-14B-Chat)
- [qwen/Qwen-14B-Chat-Int8](https://huggingface.co/qwen/Qwen-14B-Chat-Int8)
- [qwen/Qwen-14B-Chat-Int4](https://huggingface.co/qwen/Qwen-14B-Chat-Int4)

</details>

<details>

<summary>StableLM</summary>


### Quickstart

Run the following command to quickly spin up a StableLM server:

```bash
openllm start stabilityai/stablelm-tuned-alpha-3b --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any StableLM variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=stablelm) to see more StableLM-compatible models.



### Supported models

You can specify any of the following StableLM models via `openllm start`:


- [stabilityai/stablelm-tuned-alpha-3b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b)
- [stabilityai/stablelm-tuned-alpha-7b](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
- [stabilityai/stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)
- [stabilityai/stablelm-base-alpha-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)

</details>

<details>

<summary>StarCoder</summary>


### Quickstart

Run the following command to quickly spin up a StarCoder server:

```bash
openllm start bigcode/starcoder --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any StarCoder variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=starcoder) to see more StarCoder-compatible models.



### Supported models

You can specify any of the following StarCoder models via `openllm start`:


- [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)
- [bigcode/starcoderbase](https://huggingface.co/bigcode/starcoderbase)

</details>

<details>

<summary>Yi</summary>


### Quickstart

Run the following command to quickly spin up a Yi server:

```bash
openllm start 01-ai/Yi-6B --trust-remote-code
```
You can run the following code in a different terminal to interact with the server:
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```


> **Note:** Any Yi variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search=yi) to see more Yi-compatible models.



### Supported models

You can specify any of the following Yi models via `openllm start`:


- [01-ai/Yi-6B](https://huggingface.co/01-ai/Yi-6B)
- [01-ai/Yi-34B](https://huggingface.co/01-ai/Yi-34B)
- [01-ai/Yi-6B-200K](https://huggingface.co/01-ai/Yi-6B-200K)
- [01-ai/Yi-34B-200K](https://huggingface.co/01-ai/Yi-34B-200K)

</details>

<!-- update-readme.py: stop -->

More models will be integrated with OpenLLM and we welcome your contributions if you want to incorporate your custom LLMs into the ecosystem. Check out [Adding a New Model Guide](https://github.com/bentoml/OpenLLM/blob/main/ADDING_NEW_MODEL.md) to learn more.

## 📐 Quantization

Quantization is a technique to reduce the storage and computation requirements for machine learning models, particularly during inference. By approximating floating-point numbers as integers (quantized values), quantization allows for faster computations, reduced memory footprint, and can make it feasible to deploy large models on resource-constrained devices.

OpenLLM supports the following quantization techniques

- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978).
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323).
- [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629).

> [!NOTE]
> Make sure to use pre-quantized models weights when using with `openllm start`.

## ⚙️ Integrations

OpenLLM is not just a standalone product; it's a building block designed to
integrate with other powerful tools easily. We currently offer integration with
[OpenAI's Compatible Endpoints](https://platform.openai.com/docs/api-reference/completions/object),
[LlamaIndex](https://www.llamaindex.ai/),
[LangChain](https://github.com/hwchase17/langchain).

### OpenAI Compatible Endpoints

OpenLLM Server can be used as a drop-in replacement for OpenAI's API. Simply
specify the base_url to `llm-endpoint/v1` and you are good to go:

```python
import openai

client = openai.OpenAI(base_url='http://localhost:3000/v1', api_key='na')  # Here the server is running on 0.0.0.0:3000

completions = client.chat.completions.create(
  prompt='Write me a tag line for an ice cream shop.', model=model, max_tokens=64, stream=stream
)
```

The compatible endpoints supports `/chat/completions`, and `/models`

> [!NOTE]
> You can find out OpenAI example clients under the
> [examples](https://github.com/bentoml/OpenLLM/tree/main/examples) folder.

### [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/openllm/)

You can use `llama_index.llms.openllm.OpenLLMAPI` to interact with a LLM running server:

```python
from llama_index.llms.openllm import OpenLLMAPI
```

> [!NOTE]
> All synchronous and asynchronous API from `llama_index.llms.OpenLLMAPI` are supported.
> Make sure to install `llama-index-integrations-llm-openllm` to use the supported class.

### [LangChain](https://python.langchain.com/docs/integrations/llms/openllm/)

Spin up an OpenLLM server, and connect to it by specifying its URL:

```python
from langchain.llms import OpenLLMAPI

llm = OpenLLMAPI(server_url='http://44.23.123.1:3000')
llm.invoke('What is the difference between a duck and a goose? And why there are so many Goose in Canada?')

# streaming
for it in llm.stream('What is the difference between a duck and a goose? And why there are so many Goose in Canada?'):
  print(it, flush=True, end='')

# async context
await llm.ainvoke('What is the difference between a duck and a goose? And why there are so many Goose in Canada?')

# async streaming
async for it in llm.astream('What is the difference between a duck and a goose? And why there are so many Goose in Canada?'):
  print(it, flush=True, end='')
```

<!-- hatch-fancy-pypi-readme interim stop -->

<!-- hatch-fancy-pypi-readme meta start -->

## 🚀 Deploying models to production

There are several ways to deploy your LLMs:

### 🐳 Docker container

1. **Building a Bento**: With OpenLLM, you can easily build a Bento for a
   specific model, like `mistralai/Mistral-7B-Instruct-v0.1`, using the `build` command.:

   ```bash
   openllm build mistralai/Mistral-7B-Instruct-v0.1
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

Deploy OpenLLM with [BentoCloud](https://www.bentoml.com/), the inference platform
for fast moving AI teams.

1. **Create a BentoCloud account:** [sign up here](https://bentoml.com/)

2. **Log into your BentoCloud account:**

   ```bash
   bentoml cloud login --api-token <your-api-token> --endpoint <bento-cloud-endpoint>
   ```

> [!NOTE]
> Replace `<your-api-token>` and `<bento-cloud-endpoint>` with your
> specific API token and the BentoCloud endpoint respectively.

3. **Bulding a Bento**: With OpenLLM, you can easily build a Bento for a
   specific model, such as `mistralai/Mistral-7B-Instruct-v0.1`:

   ```bash
   openllm build mistralai/Mistral-7B-Instruct-v0.1
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
