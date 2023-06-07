<div align="center">
    <h1 align="center">OpenLLM</h1>
    <strong>Build, fine-tune, serve, and deploy Large-Language Models including popular ones like StableLM, Llama, Dolly, Flan-T5, Vicuna, or even your custom LLMs.<br></strong>
    <i>Powered by BentoML 🍱</i>
    <br>
</div>

#####
[![pypi_status](https://img.shields.io/pypi/v/openllm.svg)](https://pypi.org/project/openllm)
[![ci](https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml)
[![Discord](https://badgen.net/badge/icon/Discord?icon=discord&label=Join%20Us)](https://discord.gg/qc3RekjtuY)
[![Twitter](https://badgen.net/badge/icon/@bentomlai?icon=twitter&label=Follow%20Us)](https://twitter.com/bentomlai)


## 📖 Introduction

With OpenLLM, you can easily run inference with any open-source large-language
models(LLMs) and build production-ready LLM apps, powered by BentoML. Here are
some key features:

🚂 **SOTA LLMs**: With a single click, access support for state-of-the-art LLMs,
including StableLM, Llama, Alpaca, Dolly, Flan-T5, ChatGLM, Falcon, and more.

🔥 **Easy-to-use APIs**: We provide intuitive interfaces by integrating with
popular tools like BentoML, HuggingFace, LangChain, and more.

📦 **Fine-tuning your own LLM**: Customize any LLM to suit your needs with
`LLM.tuning()`. (Work In Progress)

⛓️ **Interoperability**: First-class support for LangChain and BentoML’s runner
architecture, allows easy chaining of LLMs on multiple GPUs/Nodes. (Work In Progress)

🎯 **Streamline Production Deployment**: Seamlessly package into a Bento with
`openllm build`, containerized into OCI Images, and deploy with a single click
using [☁️ BentoCloud](https://l.bentoml.com/bento-cloud).

## 🏃‍ Getting Started

To use OpenLLM, you need to have Python 3.8 (or newer) and `pip` installed on
your system. We highly recommend using a Virtual Environment to prevent package
conflicts.

You can install OpenLLM using pip as follows:

```bash
pip install openllm
```

To verify if it's installed correctly, run:

```
openllm -h
```

The correct output will be:

```
Usage: openllm [OPTIONS] COMMAND [ARGS]...

   ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
  ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝

  OpenLLM: Your one stop-and-go-solution for serving any Open Large-Language Model

      - StableLM, Falcon, ChatGLM, Dolly, Flan-T5, and more

      - Powered by BentoML 🍱
```

### Starting an LLM Server

To start an LLM server, use `openllm start`. For example, to start a `dolly-v2`
server:

```bash
openllm start dolly-v2
```

Following this, a swagger UI will be accessible at http://0.0.0.0:3000 where you
can experiment with the endpoints and sample prompts.

OpenLLM provides a built-in Python client, allowing you to interact with the
model. In a different terminal window or a Jupyter notebook, create a client to
start interacting with the model:

```python
>>> import openllm
>>> client = openllm.client.HTTPClient('http://localhost:3000')
>>> client.query('Explain to me the difference between "further" and "farther"')
```

## 🚀 Deploying to Production

To deploy your LLMs into production:

1. **Building a Bento**: With OpenLLM, you can easily build a
   Bento for a specific model, like `dolly-v2`, using the `build`
   command.:

    ```bash
    openllm build dolly-v2
    ```

    A [Bento](https://docs.bentoml.org/en/latest/concepts/bento.html#what-is-a-bento), in BentoML, is the unit of distribution. It packages your program's source code, models, files, artifacts, and dependencies.

    > _NOTE_: If you wish to build OpenLLM from the git source, set
    > `OPENLLM_DEV_BUILD=True` to include the generated wheels in the bundle.

2. **Containerize your Bento**

    ```
    bentoml containerize <name:version>
    ```

    BentoML offers a comprehensive set of options for deploying and hosting online
    ML services in production. To learn more, check out the
    [Deploying a Bento](https://docs.bentoml.org/en/latest/concepts/deploy.html)
    guide.

## 🧩 Models and Dependencies

OpenLLM currently supports the following:

- [dolly-v2](https://github.com/databrickslabs/dolly)
- [flan-t5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [chatglm](https://github.com/THUDM/ChatGLM-6B)
- [falcon](https://falconllm.tii.ae/)
- [starcoder](https://github.com/bigcode-project/starcoder)

### Model-specific Dependencies

We respect your system's space and efficiency. That's why we don't force users
to install dependencies for all models. By default, you can run `dolly-v2` and
`flan-t5` without installing any additional packages.

To enable support for a specific model, you'll need to install its corresponding
dependencies. You can do this by using `pip install "openllm[model_name]"`. For
example, to use **chatglm**:

```bash
pip install "openllm[chatglm]"
```

This will install `cpm_kernels` and `sentencepiece` additionally

### Runtime Implementations

Different LLMs may have multiple runtime implementations. For instance, they
might use Pytorch (`pt`), Tensorflow (`tf`), or Flax (`flax`).

If you wish to specify a particular runtime for a model, you can do so by
setting the `OPENLLM_{MODEL_NAME}_FRAMEWORK={runtime}` environment variable
before running `openllm start`.

For example, if you want to use the Tensorflow (`tf`) implementation for the
`flan-t5` model, you can use the following command:

```bash
OPENLLM_FLAN_T5_FRAMEWORK=tf openllm start flan-t5
```

### Integrating a New Model

OpenLLM encourages contributions by welcoming users to incorporate their custom
LLMs into the ecosystem. Checkout
[Adding a New Model Guide](https://github.com/bentoml/OpenLLM/blob/main/ADDING_NEW_MODEL.md)
to see how you can do it yourself.

## ⚙️ Integrations
OpenLLM is not just a standalone product; it's a building block designed to easily integrate with other powerful tools. We currently offer integration with [BentoML](https://github.com/bentoml/BentoML) and [LangChain](https://github.com/hwchase17/langchain).

### BentoML
OpenLLM models can be integrated as a [Runner](https://docs.bentoml.org/en/latest/concepts/runner.html) in your BentoML service. These runners has a `generate` method that takes a string as a prompt and returns a corresponding output string. This will allow you to plug and play any OpenLLM models with your existing ML workflow.

```python
import bentoml
import openllm

model = "dolly-v2"

llm_config = openllm.AutoConfig.for_model(model)
llm_runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(
    name=f"llm-dolly-v2-service", runners=[llm_runner]
)

@svc.api(input=Text(), output=Text())
async def prompt(input_text: str) -> str:
    answer = await llm_runner.generate(input_text)
    return answer
```

### LangChain (⏳Coming Soon!)
In future LangChain releases, you'll be able to effortlessly invoke OpenLLM models, like so:
```python
from langchain.llms import OpenLLM
llm = OpenLLM.for_model(model_name='flan-t5')
llm("What is the difference between a duck and a goose?")
```
 if you have an OpenLLM server deployed elsewhere, you can connect to it by specifying its URL:

```python
from langchain.llms import OpenLLM
llm = OpenLLM.for_model(server_url='http://localhost:8000', server_type='http')
llm("What is the difference between a duck and a goose?")
```


## 🍇 Telemetry

OpenLLM collects usage data to enhance user experience and improve the product.
We only report OpenLLM's internal API calls and ensure maximum privacy by
excluding sensitive information. We will never collect user code, model data, or
stack traces. For usage tracking, check out the
[code](./src/openllm/utils/analytics.py).

You can opt-out of usage tracking by using the `--do-not-track` CLI option:

```bash
openllm [command] --do-not-track
```

Or by setting environment variable `OPENLLM_DO_NOT_TRACK=True`:

```bash
export OPENLLM_DO_NOT_TRACK=True
```

## 👥 Community

Engage with like-minded individuals passionate about LLMs, AI, and more on our
[Discord](https://l.bentoml.com/join-openllm-discord)!

OpenLLM is actively maintained by the BentoML team. Feel free to reach out and
join us in our pursuit to make LLMs more accessible and easy-to-use👉
[Join our Slack community!](https://l.bentoml.com/join-slack)

## 🎁 Contributing

We welcome contributions! If you're interested in enhancing OpenLLM's
capabilities or have any questions, don't hesitate to reach out in our
[discord channel](https://l.bentoml.com/join-openllm-discord).

Checkout our
[Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md)
if you wish to contribute to OpenLLM's codebase.
