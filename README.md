<div align="center">
    <h1 align="center">OpenLLM</h1>
    <br>
    <strong>REST/gRPC API server for running any Open Large-Language Model - StableLM, Llama, Alpaca, Dolly, Flan-T5, and more<br></strong>
    <i>Powered by BentoML 🍱 + HuggingFace 🤗</i>
    <br>
</div>

<br/>

To get started, simply install OpenLLM with pip:

```bash
pip install openllm
```

To start a LLM server, `openllm start` allows you to start any supported LLM
with a single command. For example, to start a `dolly-v2` server:

## 😌 tl;dr?

```bash
openllm start dolly-v2

# Starting LLM Server for 'dolly_v2'
#
# 2023-05-27T04:55:36-0700 [INFO] [cli] Environ for worker 0: set CPU thread coun t to 10
# 2023-05-27T04:55:36-0700 [INFO] [cli] Prometheus metrics for HTTP BentoServer f rom "_service.py:svc" can be accessed at http://localhost:3000/metrics.
# 2023-05-27T04:55:36-0700 [INFO] [cli] Starting production HTTP BentoServer from "_service.py:svc" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

To see a list of supported LLMs, run `openllm start --help`.

On a different terminal window, open a IPython session and create a client to
start interacting with the model:

```python
>>> import openllm
>>> client = openllm.client.HTTPClient('http://localhost:3000')
>>> client.query('Explain to me the difference between "further" and "farther"')
```

To package the LLM into a Bento, simply use `openllm build`:

```bash
openllm build dolly-v2
```

> NOTE: To build OpenLLM from git source, pass in `OPENLLM_DEV_BUILD=True` to
> include the generated wheels into the bundle.

To fine-tune your own LLM, either use `LLM.tuning()`:

```python
>>> import openllm
>>> flan_t5 = openllm.LLM.from_pretrained("flan-t5")
>>> def fine_tuning():
...     fined_tune = flan_t5.tuning(method=openllm.tune.LORA | openllm.tune.P_TUNING, dataset='wikitext-2', ...)
...     fined_tune.save_pretrained('./fine-tuned-flan-t5', version='wikitext')
...     return fined_tune.path  # get the path of the pretrained
>>> finetune_path = fine_tuning()
>>> fined_tune_flan_t5 = openllm.LLM.from_pretrained('flan-t5', pretrained=finetune_path)
>>> fined_tune_flan_t5.generate('Explain to me the difference between "further" and "farther"')
```

## 📚 Features

🚂 **SOTA LLMs**: One-click stop-and-go supports for state-of-the-art LLMs,
including StableLM, Llama, Alpaca, Dolly, Flan-T5, ChatGLM, Falcon, and more.

📦 **Fine-tuning your own LLM**: Easily fine-tune any LLM with `LLM.tuning()`.

🔥 **BentoML 🤝 HuggingFace**: Built on top of BentoML and HuggingFace's
ecosystem (transformers, optimum, peft, accelerate, datasets), provides similar
APIs for ease-of-use.

⛓️ **Interoperability**: First class support for LangChain and
[🤗 Hub](https://huggingface.co/) allows you to easily chain LLMs together.

🎯 **Streamline production deployment**: Easily deploy any LLM via
`openllm bundle` with the following:

- [☁️ BentoML Cloud](https://l.bentoml.com/bento-cloud): the fastest way to
  deploy your bento, simple and at scale
- [🦄️ Yatai](https://github.com/bentoml/yatai): Model Deployment at scale on
  Kubernetes
- [🚀 bentoctl](https://github.com/bentoml/bentoctl): Fast model deployment on
  AWS SageMaker, Lambda, ECE, GCP, Azure, Heroku, and more!

## 🍇 Telemetry

OpenLLM collects usage data that helps the team to improve the product. Only
OpenLLM's internal API calls are being reported. We strip out as much
potentially sensitive information as possible, and we will never collect user
code, model data, or stack traces. Here's the
[code](./src/openllm/utils/analytics.py) for usage tracking. You can opt-out of
usage tracking by the `--do-not-track` CLI option:

```bash
openllm [command] --do-not-track
```

Or by setting environment variable `OPENLLM_DO_NOT_TRACK=True`:

```bash
export OPENLLM_DO_NOT_TRACK=True
```
