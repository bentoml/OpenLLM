<div align="center">
    <h1 align="center">OpenLLM</h1>
    <br>
    <strong>REST/gRPC API server for running any Open Large-Language Model - StableLM, Llama, Alpaca, Dolly, Flan-T5, and more<br></strong>
    <i>Powered by BentoML 🍱 + HuggingFace 🤗</i>
    <br>
</div>

## 📖 Introduction

Welcome to OpenLLM, a robust platform designed to streamline the usage of large language models (LLMs). Here are some key features:

🚂 **SOTA LLMs**: With a single click, access support for state-of-the-art LLMs, including StableLM, Llama, Alpaca, Dolly, Flan-T5, ChatGLM, Falcon, and more.

🔥 **BentoML 🤝 HuggingFace**: Leveraging the power of BentoML and HuggingFace's ecosystem (transformers, optimum, peft, accelerate, datasets), OpenLLM offers user-friendly APIs for seamless integration and usage.

📦 **Fine-tuning your own LLM**: Customize any LLM to suit your needs with `LLM.tuning()`. (Work In Progress)

⛓️ **Interoperability**: Our first-class support for LangChain and [🤗 Hub](https://huggingface.co/) allows easy chaining of LLMs. (Work In Progress)

🎯 **Streamline production deployment**: Deploy any LLM effortlessly using `openllm bundle` with [☁️ BentoML Cloud](https://l.bentoml.com/bento-cloud).

## 🏃‍ Getting Started
To use OpenLLM, you need to have Python 3.8 (or newer) and `pip` installed on your system. We highly recommend using a Virtual Environment to prevent package conflicts.

You can install OpenLLM using pip as follows:

```bash
pip install openllm
```
To verify if it's installed correctly, run:
```
openllm version
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

      - Powered by BentoML 🍱 + HuggingFace 🤗
```

### Starting an LLM Server
To start an LLM server, use `openllm start`. For example, to start a `dolly-v2` server:

```bash
openllm start dolly-v2
```
Following this, a swagger UI will be accessible at http://0.0.0.0:3000 where you can experiment with the endpoints and sample prompts.

OpenLLM provides a built-in Python client, allowing you to interact with the model. In a different terminal window or a Jupyter notebook, create a client to start interacting with the model:

```python
>>> import openllm
>>> client = openllm.client.HTTPClient('http://localhost:3000')
>>> client.query('Explain to me the difference between "further" and "farther"')
```
## 🚀 Deploying to Production 
Take your Large Language Models (LLMs) from experimentation to production effortlessly with OpenLLM and [BentoCloud](https://www.bentoml.com/bento-cloud/). These are the steps:

1. **Create a BentoCloud Account**: If you haven't already, start by signing up for a [BentoCloud](https://www.bentoml.com/bento-cloud/) account.

2. **Login to BentoCloud**: Once you've created your account, authenticate your local environment with your BentoCloud account. Use the command below, replacing `<your-api-token>` and `<bento-cloud-endpoint>` with your specific API token and the BentoCloud endpoint respectively:

```bash
bentoml yatai login --api-token <your-api-token> --endpoint <bento-cloud-endpoint>
```

3. **Build Your BentoML Service**: With OpenLLM, you can easily build your BentoML service for a specific model, like `dolly-v2`, using the `bundle` command:

```bash
openllm bundle dolly-v2
```

> _NOTE_: If you wish to build OpenLLM from the git source, set `OPENLLM_DEV_BUILD=True` to include the generated wheels in the bundle.

4. **Push Your Service to BentoCloud**: Once you've built your BentoML service, it's time to push it to BentoCloud. Use the `push` command and replace `<name:version>` with your service's name and version:

```bash
bentoml push <name:version>
```

BentoML offers a comprehensive set of options for deploying and hosting online ML services in production. To learn more, check out the [Deploying a Bento](https://docs.bentoml.org/en/latest/concepts/deploy.html) guide.

## 🧩  Models and Dependencies
OpenLLM currently supports the following:
* [dolly-v2](https://github.com/databrickslabs/dolly)
* [flan-t5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
* [chatglm](https://github.com/THUDM/ChatGLM-6B)
* [falcon](https://falconllm.tii.ae/)
* [starcoder](https://github.com/bigcode-project/starcoder)

### Model-specific Dependencies
We respect your system's space and efficiency. That's why we don't force users to install dependencies for all models. By default, you can run `dolly-v2` and `flan-t5` without installing any additional packages.

To enable support for a specific model, you'll need to install its corresponding dependencies. You can do this by using `pip install openllm[model_name]`. For example, to use **chatglm**:

```bash
pip install openllm[chatglm]
```
This will install `cpm_kernels` and `sentencepiece` additionally

### Runtime Implementations

Different LLMs may have multiple runtime implementations. For instance, they might use Pytorch (`pt`), Tensorflow (`tf`), or Flax (`flax`).

If you wish to specify a particular runtime for a model, you can do so by setting the `OPENLLM_{MODEL_NAME}_FRAMEWORK={runtime}` environment variable before running `openllm start`.

For example, if you want to use the Tensorflow (`tf`) implementation for the `flan-t5` model, you can use the following command:

```bash
OPENLLM_FLAN_T5_FRAMEWORK=tf openllm start flan-t5
```

## 🍇 Telemetry

OpenLLM collects usage data to enhance user experience and improve the product. We only report OpenLLM's internal API calls and ensure maximum privacy by excluding sensitive information. We will never collect user code, model data, or stack traces. For usage tracking, check out the [code](./src/openllm/utils/analytics.py).

You can opt-out of usage tracking by using the `--do-not-track` CLI option:

```bash
openllm [command] --do-not-track
```

Or by setting environment variable `OPENLLM_DO_NOT_TRACK=True`:

```bash
export OPENLLM_DO_NOT_TRACK=True
```

## 👥 Community
Engage with like-minded individuals passionate about LLMs, AI, and more on our [Discord](https://l.bentoml.com/join-openllm-discord)!

OpenLLM is actively maintained by the BentoML team. Feel free to reach out and join us in our pursuit to make LLMs more accessible and easy-to-use👉 [Join our Slack community!](https://l.bentoml.com/join-slack)


## 🎁 Contributing
We welcome contributions! If you're interested in enhancing OpenLLM's capabilities or have any questions, don't hesitate to reach out in our [discord channel](https://l.bentoml.com/join-openllm-discord).

Checkout our [Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md) if you wish to contribute to OpenLLM's codebase.
