<div align="center">
    <h1 align="center">OpenLLM</h1>
    <br>
    <strong>REST/gRPC API server for running any Open Large-Language Model - StableLM, Llama, Alpaca, Dolly, Flan-T5, and more<br></strong>
    <i>Powered by BentoML 🍱 + HuggingFace 🤗</i>
    <br>
</div>

To get started, simply install OpenLLM with pip:

```bash
pip install openllm
```

> NOTE: Currently, OpenLLM is built with pydantic v2. At the time of writing,
> Pydantic v2 is still in alpha stage. To get pydantic v2, do
> `pip install -U --pre pydantic`

To start a LLM server, `openllm start` allows you to start any supported LLM
with a single command. For example, to start a `dolly-v2` server:

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

🎯 To streamline production deployment, you can use the following:

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
