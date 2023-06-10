<div align="center">
    <h1 align="center">🦾 OpenLLM</h1>
    <a href="https://pypi.org/project/openllm">
        <img src="https://img.shields.io/pypi/v/openllm.svg" alt="pypi_status" />
    </a><a href="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml">
        <img src="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml/badge.svg?branch=main" alt="ci" />
    </a><a href="https://twitter.com/bentomlai">
        <img src="https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow%20Us" alt="Twitter" />
    </a><a href="https://l.bentoml.com/join-openllm-discord">
        <img src="https://badgen.net/badge/icon/OpenLLM/7289da?icon=discord&label=Join%20Us" alt="Discord" />
    </a><br>
    <p>An open platform for operating large language models(LLMs) in production.</br>
    Fine-tune, serve, deploy, and monitor any LLMs with ease.</p>
    <i></i>
</div>

<br/>

## 📖 Introduction

With OpenLLM, you can run inference with any open-source large-language models(LLMs),
deploy to the cloud or on-premises, and build powerful AI apps.

🚂 **SOTA LLMs**: built-in supports a wide range of open-source LLMs and model runtime,
including StableLM, Falcon, Dolly, Flan-T5, ChatGLM, StarCoder and more.

🔥 **Flexible APIs**: serve LLMs over RESTful API or gRPC with one command, query
via WebUI, CLI, our Python/Javascript client, or any HTTP client.

⛓️ **Freedom To Build**: First-class support for LangChain and BentoML allows you to
easily create your own AI apps by composing LLMs with other models and services.

🎯 **Streamline Deployment**: build your LLM server into docker Images or
deploy as serverless endpoint via [☁️ BentoCloud](https://l.bentoml.com/bento-cloud).

🤖️ **Bring your own LLM**: Fine-tune any LLM to suit your needs with
`LLM.tuning()`. (Coming soon)


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
$ openllm -h

Usage: openllm [OPTIONS] COMMAND [ARGS]...

   ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
  ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝

  An open platform for operating large language models in production.
  Fine-tune, serve, deploy, and monitor any LLMs with ease.
```

### Starting an LLM Server

To start an LLM server, use `openllm start`. For example, to start a `dolly-v2`
server:

```bash
openllm start dolly-v2
```

Following this, a Web UI will be accessible at http://0.0.0.0:3000 where you
can experiment with the endpoints and sample input prompts.

OpenLLM provides a built-in Python client, allowing you to interact with the
model. In a different terminal window or a Jupyter notebook, create a client to
start interacting with the model:

```python
>>> import openllm
>>> client = openllm.client.HTTPClient('http://localhost:3000')
>>> client.query('Explain to me the difference between "further" and "farther"')
```

You can also use the `openllm query` command to query the model from the
terminal:

```bash
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'Explain to me the difference between "further" and "farther"'
```

Visit `http://0.0.0.0:3000/docs.json` for OpenLLM's API specification.


## 🧩 Supported Models

The following models are currently supported in OpenLLM. By default, OpenLLM doesn't
include dependencies to run all models. The extra model-specific dependencies can be 
installed with the instructions below:

<!-- update-readme.py: start -->

| Model                                                                 | CPU | GPU | Installation                           |
| --------------------------------------------------------------------- | --- | --- | ---------------------------------- |
| [flan-t5](https://huggingface.co/docs/transformers/model_doc/flan-t5) | ✅  | ✅  | `pip install "openllm[flan-t5]"`   |
| [dolly-v2](https://github.com/databrickslabs/dolly)                   | ✅  | ✅  | `pip install openllm`             |
| [chatglm](https://github.com/THUDM/ChatGLM-6B)                        | ❌  | ✅  | `pip install "openllm[chatglm]"`   |
| [starcoder](https://github.com/bigcode-project/starcoder)             | ❌  | ✅  | `pip install "openllm[starcoder]"` |
| [falcon](https://falconllm.tii.ae/)                                   | ❌  | ✅  | `pip install "openllm[falcon]"`    |
| [stablelm](https://github.com/Stability-AI/StableLM)                  | ❌  | ✅  | `pip install openllm`             |

<!-- update-readme.py: stop -->

### Runtime Implementations (Experimental)

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
LLMs into the ecosystem. Check out [Adding a New Model Guide](https://github.com/bentoml/OpenLLM/blob/main/ADDING_NEW_MODEL.md)
to see how you can do it yourself.

## ⚙️ Integrations

OpenLLM is not just a standalone product; it's a building block designed to
easily integrate with other powerful tools. We currently offer integration with
[BentoML](https://github.com/bentoml/BentoML) and
[LangChain](https://github.com/hwchase17/langchain).

### BentoML

OpenLLM models can be integrated as a
[Runner](https://docs.bentoml.org/en/latest/concepts/runner.html) in your
BentoML service. These runners have a `generate` method that takes a string as a
prompt and returns a corresponding output string. This will allow you to plug
and play any OpenLLM models with your existing ML workflow.

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

In future LangChain releases, you'll be able to effortlessly invoke OpenLLM
models, like so:

```python
from langchain.llms import OpenLLM
llm = OpenLLM.for_model(model_name='flan-t5')
llm("What is the difference between a duck and a goose?")
```

if you have an OpenLLM server deployed elsewhere, you can connect to it by
specifying its URL:

```python
from langchain.llms import OpenLLM
llm = OpenLLM.for_model(server_url='http://localhost:8000', server_type='http')
llm("What is the difference between a duck and a goose?")
```

## 🚀 Deploying to Production

To deploy your LLMs into production:

1. **Building a Bento**: With OpenLLM, you can easily build a Bento for a
   specific model, like `dolly-v2`, using the `build` command.:

   ```bash
   openllm build dolly-v2
   ```

   A [Bento](https://docs.bentoml.org/en/latest/concepts/bento.html#what-is-a-bento),
   in BentoML, is the unit of distribution. It packages your program's source
   code, models, files, artifacts, and dependencies.


2. **Containerize your Bento**

   ```
   bentoml containerize <name:version>
   ```

   BentoML offers a comprehensive set of options for deploying and hosting
   online ML services in production. To learn more, check out the
   [Deploying a Bento](https://docs.bentoml.org/en/latest/concepts/deploy.html)
   guide.


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
