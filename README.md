![Banner for OpenLLM](/assets/main-banner.png)

<!-- hatch-fancy-pypi-readme intro start -->

<div align="center">
    <h1 align="center">🦾 OpenLLM</h1>
    <a href="https://pypi.org/project/openllm">
        <img src="https://img.shields.io/pypi/v/openllm.svg?logo=pypi&label=PyPI&logoColor=gold" alt="pypi_status" />
    </a><a href="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml">
        <img src="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml/badge.svg?branch=main" alt="ci" />
    </a><a href="https://twitter.com/bentomlai">
        <img src="https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow%20Us" alt="Twitter" />
    </a><a href="https://l.bentoml.com/join-openllm-discord">
        <img src="https://badgen.net/badge/icon/OpenLLM/7289da?icon=discord&label=Join%20Us" alt="Discord" />
    </a><br>
    </a><a href="https://pypi.org/project/openllm">
        <img src="https://img.shields.io/pypi/pyversions/openllm.svg?logo=python&label=Python&logoColor=gold" alt="python_version" />
    </a><a href="https://github.com/pypa/hatch">
        <img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" alt="Hatch" />
    </a><br>
    </a><a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Ruff" />
    </a><br>
    <p>An open platform for operating large language models (LLMs) in production.</br>
    Fine-tune, serve, deploy, and monitor any LLMs with ease.</p>
    <i></i>
</div>

## 📖 Introduction

With OpenLLM, you can run inference with any open-source large-language models,
deploy to the cloud or on-premises, and build powerful AI apps.

🚂 **State-of-the-art LLMs**: built-in supports a wide range of open-source LLMs
and model runtime, including Llama 2，StableLM, Falcon, Dolly, Flan-T5, ChatGLM,
StarCoder and more.

🔥 **Flexible APIs**: serve LLMs over RESTful API or gRPC with one command,
query via WebUI, CLI, our Python/Javascript client, or any HTTP client.

⛓️ **Freedom To Build**: First-class support for LangChain, BentoML and Hugging
Face that allows you to easily create your own AI apps by composing LLMs with
other models and services.

🎯 **Streamline Deployment**: Automatically generate your LLM server Docker
Images or deploy as serverless endpoint via
[☁️ BentoCloud](https://l.bentoml.com/bento-cloud).

🤖️ **Bring your own LLM**: Fine-tune any LLM to suit your needs with
`LLM.tuning()`. (Coming soon)

<!-- hatch-fancy-pypi-readme intro stop -->

![Gif showing OpenLLM Intro](/assets/output.gif)

<br/>

<!-- hatch-fancy-pypi-readme interim start -->

## 🏃 Getting Started

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

To start an LLM server, use `openllm start`. For example, to start a
[`OPT`](https://huggingface.co/docs/transformers/model_doc/opt) server, do the
following:

```bash
openllm start opt
```

Following this, a Web UI will be accessible at http://localhost:3000 where you
can experiment with the endpoints and sample input prompts.

OpenLLM provides a built-in Python client, allowing you to interact with the
model. In a different terminal window or a Jupyter Notebook, create a client to
start interacting with the model:

```python
import openllm
client = openllm.client.HTTPClient('http://localhost:3000')
client.query('Explain to me the difference between "further" and "farther"')
```

You can also use the `openllm query` command to query the model from the
terminal:

```bash
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'Explain to me the difference between "further" and "farther"'
```

Visit `http://localhost:3000/docs.json` for OpenLLM's API specification.

OpenLLM seamlessly supports many models and their variants.
Users can also specify different variants of the model to be served, by
providing the `--model-id` argument, e.g.:

```bash
openllm start flan-t5 --model-id google/flan-t5-large
```

> **Note** that `openllm` also supports all variants of fine-tuning weights, custom model path
> as well as quantized weights for any of the supported models as long as it can be loaded with
> the model architecture. Refer to [supported models](https://github.com/bentoml/OpenLLM/tree/main#-supported-models) section for models' architecture.

Use the `openllm models` command to see the list of models and their variants
supported in OpenLLM.

## 🧩 Supported Models

The following models are currently supported in OpenLLM. By default, OpenLLM
doesn't include dependencies to run all models. The extra model-specific
dependencies can be installed with the instructions below:

<!-- update-readme.py: start -->

<table align='center'>
<tr>
<th>Model</th>
<th>Architecture</th>
<th>Model Ids</th>
<th>Installation</th>
</tr>
<tr>

<td><a href=https://github.com/THUDM/ChatGLM-6B>chatglm</a></td>
<td><a href=https://github.com/THUDM/ChatGLM-6B><code>ChatGLMForConditionalGeneration</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/thudm/chatglm-6b><code>thudm/chatglm-6b</code></a></li>
<li><a href=https://huggingface.co/thudm/chatglm-6b-int8><code>thudm/chatglm-6b-int8</code></a></li>
<li><a href=https://huggingface.co/thudm/chatglm-6b-int4><code>thudm/chatglm-6b-int4</code></a></li>
<li><a href=https://huggingface.co/thudm/chatglm2-6b><code>thudm/chatglm2-6b</code></a></li>
<li><a href=https://huggingface.co/thudm/chatglm2-6b-int4><code>thudm/chatglm2-6b-int4</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[chatglm]"
```

</td>
</tr>
<tr>

<td><a href=https://github.com/databrickslabs/dolly>dolly-v2</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM><code>GPTNeoXForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/databricks/dolly-v2-3b><code>databricks/dolly-v2-3b</code></a></li>
<li><a href=https://huggingface.co/databricks/dolly-v2-7b><code>databricks/dolly-v2-7b</code></a></li>
<li><a href=https://huggingface.co/databricks/dolly-v2-12b><code>databricks/dolly-v2-12b</code></a></li></ul>

</td>
<td>

```bash
pip install openllm
```

</td>
</tr>
<tr>

<td><a href=https://falconllm.tii.ae/>falcon</a></td>
<td><a href=https://falconllm.tii.ae/><code>FalconForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/tiiuae/falcon-7b><code>tiiuae/falcon-7b</code></a></li>
<li><a href=https://huggingface.co/tiiuae/falcon-40b><code>tiiuae/falcon-40b</code></a></li>
<li><a href=https://huggingface.co/tiiuae/falcon-7b-instruct><code>tiiuae/falcon-7b-instruct</code></a></li>
<li><a href=https://huggingface.co/tiiuae/falcon-40b-instruct><code>tiiuae/falcon-40b-instruct</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[falcon]"
```

</td>
</tr>
<tr>

<td><a href=https://huggingface.co/docs/transformers/model_doc/flan-t5>flan-t5</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/t5#transformers.T5ForConditionalGeneration><code>T5ForConditionalGeneration</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/google/flan-t5-small><code>google/flan-t5-small</code></a></li>
<li><a href=https://huggingface.co/google/flan-t5-base><code>google/flan-t5-base</code></a></li>
<li><a href=https://huggingface.co/google/flan-t5-large><code>google/flan-t5-large</code></a></li>
<li><a href=https://huggingface.co/google/flan-t5-xl><code>google/flan-t5-xl</code></a></li>
<li><a href=https://huggingface.co/google/flan-t5-xxl><code>google/flan-t5-xxl</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[flan-t5]"
```

</td>
</tr>
<tr>

<td><a href=https://github.com/EleutherAI/gpt-neox>gpt-neox</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM><code>GPTNeoXForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/eleutherai/gpt-neox-20b><code>eleutherai/gpt-neox-20b</code></a></li></ul>

</td>
<td>

```bash
pip install openllm
```

</td>
</tr>
<tr>

<td><a href=https://github.com/facebookresearch/llama>llama</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaForCausalLM><code>LlamaForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/meta-llama/Llama-2-70b-chat-hf><code>meta-llama/Llama-2-70b-chat-hf</code></a></li>
<li><a href=https://huggingface.co/meta-llama/Llama-2-13b-chat-hf><code>meta-llama/Llama-2-13b-chat-hf</code></a></li>
<li><a href=https://huggingface.co/meta-llama/Llama-2-7b-chat-hf><code>meta-llama/Llama-2-7b-chat-hf</code></a></li>
<li><a href=https://huggingface.co/meta-llama/Llama-2-70b-hf><code>meta-llama/Llama-2-70b-hf</code></a></li>
<li><a href=https://huggingface.co/meta-llama/Llama-2-13b-hf><code>meta-llama/Llama-2-13b-hf</code></a></li>
<li><a href=https://huggingface.co/meta-llama/Llama-2-7b-hf><code>meta-llama/Llama-2-7b-hf</code></a></li>
<li><a href=https://huggingface.co/NousResearch/llama-2-70b-chat-hf><code>NousResearch/llama-2-70b-chat-hf</code></a></li>
<li><a href=https://huggingface.co/NousResearch/llama-2-13b-chat-hf><code>NousResearch/llama-2-13b-chat-hf</code></a></li>
<li><a href=https://huggingface.co/NousResearch/llama-2-7b-chat-hf><code>NousResearch/llama-2-7b-chat-hf</code></a></li>
<li><a href=https://huggingface.co/NousResearch/llama-2-70b-hf><code>NousResearch/llama-2-70b-hf</code></a></li>
<li><a href=https://huggingface.co/NousResearch/llama-2-13b-hf><code>NousResearch/llama-2-13b-hf</code></a></li>
<li><a href=https://huggingface.co/NousResearch/llama-2-7b-hf><code>NousResearch/llama-2-7b-hf</code></a></li>
<li><a href=https://huggingface.co/openlm-research/open_llama_7b_v2><code>openlm-research/open_llama_7b_v2</code></a></li>
<li><a href=https://huggingface.co/openlm-research/open_llama_3b_v2><code>openlm-research/open_llama_3b_v2</code></a></li>
<li><a href=https://huggingface.co/openlm-research/open_llama_13b><code>openlm-research/open_llama_13b</code></a></li>
<li><a href=https://huggingface.co/huggyllama/llama-65b><code>huggyllama/llama-65b</code></a></li>
<li><a href=https://huggingface.co/huggyllama/llama-30b><code>huggyllama/llama-30b</code></a></li>
<li><a href=https://huggingface.co/huggyllama/llama-13b><code>huggyllama/llama-13b</code></a></li>
<li><a href=https://huggingface.co/huggyllama/llama-7b><code>huggyllama/llama-7b</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[llama]"
```

</td>
</tr>
<tr>

<td><a href=https://huggingface.co/mosaicml>mpt</a></td>
<td><a href=https://huggingface.co/mosaicml><code>MPTForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/mosaicml/mpt-7b><code>mosaicml/mpt-7b</code></a></li>
<li><a href=https://huggingface.co/mosaicml/mpt-7b-instruct><code>mosaicml/mpt-7b-instruct</code></a></li>
<li><a href=https://huggingface.co/mosaicml/mpt-7b-chat><code>mosaicml/mpt-7b-chat</code></a></li>
<li><a href=https://huggingface.co/mosaicml/mpt-7b-storywriter><code>mosaicml/mpt-7b-storywriter</code></a></li>
<li><a href=https://huggingface.co/mosaicml/mpt-30b><code>mosaicml/mpt-30b</code></a></li>
<li><a href=https://huggingface.co/mosaicml/mpt-30b-instruct><code>mosaicml/mpt-30b-instruct</code></a></li>
<li><a href=https://huggingface.co/mosaicml/mpt-30b-chat><code>mosaicml/mpt-30b-chat</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[mpt]"
```

</td>
</tr>
<tr>

<td><a href=https://huggingface.co/docs/transformers/model_doc/opt>opt</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/opt#transformers.OPTForCausalLM><code>OPTForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/facebook/opt-125m><code>facebook/opt-125m</code></a></li>
<li><a href=https://huggingface.co/facebook/opt-350m><code>facebook/opt-350m</code></a></li>
<li><a href=https://huggingface.co/facebook/opt-1.3b><code>facebook/opt-1.3b</code></a></li>
<li><a href=https://huggingface.co/facebook/opt-2.7b><code>facebook/opt-2.7b</code></a></li>
<li><a href=https://huggingface.co/facebook/opt-6.7b><code>facebook/opt-6.7b</code></a></li>
<li><a href=https://huggingface.co/facebook/opt-66b><code>facebook/opt-66b</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[opt]"
```

</td>
</tr>
<tr>

<td><a href=https://github.com/Stability-AI/StableLM>stablelm</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/gpt_neox#transformers.GPTNeoXForCausalLM><code>GPTNeoXForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/stabilityai/stablelm-tuned-alpha-3b><code>stabilityai/stablelm-tuned-alpha-3b</code></a></li>
<li><a href=https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b><code>stabilityai/stablelm-tuned-alpha-7b</code></a></li>
<li><a href=https://huggingface.co/stabilityai/stablelm-base-alpha-3b><code>stabilityai/stablelm-base-alpha-3b</code></a></li>
<li><a href=https://huggingface.co/stabilityai/stablelm-base-alpha-7b><code>stabilityai/stablelm-base-alpha-7b</code></a></li></ul>

</td>
<td>

```bash
pip install openllm
```

</td>
</tr>
<tr>

<td><a href=https://github.com/bigcode-project/starcoder>starcoder</a></td>
<td><a href=https://huggingface.co/docs/transformers/main/model_doc/gpt_bigcode#transformers.GPTBigCodeForCausalLM><code>GPTBigCodeForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/bigcode/starcoder><code>bigcode/starcoder</code></a></li>
<li><a href=https://huggingface.co/bigcode/starcoderbase><code>bigcode/starcoderbase</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[starcoder]"
```

</td>
</tr>
<tr>

<td><a href=https://github.com/baichuan-inc/Baichuan-7B>baichuan</a></td>
<td><a href=https://github.com/baichuan-inc/Baichuan-7B><code>BaiChuanForCausalLM</code></a></td>
<td>

<ul><li><a href=https://huggingface.co/baichuan-inc/baichuan-7b><code>baichuan-inc/baichuan-7b</code></a></li>
<li><a href=https://huggingface.co/baichuan-inc/baichuan-13b-base><code>baichuan-inc/baichuan-13b-base</code></a></li>
<li><a href=https://huggingface.co/baichuan-inc/baichuan-13b-chat><code>baichuan-inc/baichuan-13b-chat</code></a></li>
<li><a href=https://huggingface.co/fireballoon/baichuan-vicuna-chinese-7b><code>fireballoon/baichuan-vicuna-chinese-7b</code></a></li>
<li><a href=https://huggingface.co/fireballoon/baichuan-vicuna-7b><code>fireballoon/baichuan-vicuna-7b</code></a></li>
<li><a href=https://huggingface.co/hiyouga/baichuan-7b-sft><code>hiyouga/baichuan-7b-sft</code></a></li></ul>

</td>
<td>

```bash
pip install "openllm[baichuan]"
```

</td>
</tr>
</table>

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

> **Note** For GPU support on Flax, refers to
> [Jax's installation](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)
> to make sure that you have Jax support for the corresponding CUDA version.

### Quantisation

OpenLLM supports quantisation with
[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and
[GPTQ](https://arxiv.org/abs/2210.17323)

```bash
openllm start mpt --quantize int8
```

To run inference with `gptq`, simply pass `--quantize gptq`:

```bash
openllm start falcon --model-id TheBloke/falcon-40b-instruct-GPTQ --quantize gptq --device 0
```

> **Note**: to run GPTQ, make sure to install with
> `pip install "openllm[gptq]"`. The weights of all supported models should be
> quantized before serving. See
> [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) for more
> information on GPTQ quantisation.

### Fine-tuning support (Experimental)

One can serve OpenLLM models with any PEFT-compatible layers with
`--adapter-id`:

```bash
openllm start opt --model-id facebook/opt-6.7b --adapter-id aarnphm/opt-6-7b-quotes
```

It also supports adapters from custom paths:

```bash
openllm start opt --model-id facebook/opt-6.7b --adapter-id /path/to/adapters
```

To use multiple adapters, use the following format:

```bash
openllm start opt --model-id facebook/opt-6.7b --adapter-id aarnphm/opt-6.7b-lora --adapter-id aarnphm/opt-6.7b-lora:french_lora
```

By default, the first adapter-id will be the default Lora layer, but optionally
users can change what Lora layer to use for inference via `/v1/adapters`:

```bash
curl -X POST http://localhost:3000/v1/adapters --json '{"adapter_name": "vn_lora"}'
```

Note that for multiple adapter-name and adapter-id, it is recommended to update
to use the default adapter before sending the inference, to avoid any
performance degradation

To include this into the Bento, one can also provide a `--adapter-id` into
`openllm build`:

```bash
openllm build opt --model-id facebook/opt-6.7b --adapter-id ...
```

> **Note**: We will gradually roll out support for fine-tuning all models.
> The following models contain fine-tuning support: OPT, Falcon, LlaMA.

### Integrating a New Model

OpenLLM encourages contributions by welcoming users to incorporate their custom
LLMs into the ecosystem. Check out
[Adding a New Model Guide](https://github.com/bentoml/OpenLLM/blob/main/ADDING_NEW_MODEL.md)
to see how you can do it yourself.

### Embeddings

OpenLLM tentatively provides embeddings endpoint for supported models.
This can be accessed via `/v1/embeddings`.

To use via CLI, simply call ``openllm embed``:

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

To invoke this endpoints, use ``client.embed`` from the Python SDK:

```python
import openllm

client = openllm.client.HTTPClient("http://localhost:3000")

client.embed("I like to eat apples")
```

> **Note**: Currently, the following model framily supports embeddings: Llama, T5 (Flan-T5, FastChat, etc.), ChatGLM

## ⚙️ Integrations

OpenLLM is not just a standalone product; it's a building block designed to
integrate with other powerful tools easily. We currently offer integration with
[BentoML](https://github.com/bentoml/BentoML),
[LangChain](https://github.com/hwchase17/langchain),
and [Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents).

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

llm = OpenLLM(model_name="dolly-v2", model_id='databricks/dolly-v2-7b', device_map='auto')

llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
```

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

> **Note** You can find out more examples under the
> [examples](https://github.com/bentoml/OpenLLM/tree/main/examples) folder.


### Transformers Agents

OpenLLM seamlessly integrates with [Transformers Agents](https://huggingface.co/docs/transformers/transformers_agents).


> **Warning** The Transformers Agent is still at an experimental stage. It is
> recommended to install OpenLLM with `pip install -r nightly-requirements.txt` to get
> the latest API update for HuggingFace agent.

```python
import transformers

agent = transformers.HfAgent("http://localhost:3000/hf/agent")  # URL that runs the OpenLLM server

agent.run("Is the following `text` positive or negative?", text="I don't like how this models is generate inputs")
```

> **Note** Only `starcoder` is currently supported with Agent integration. The
> example above was also ran with four T4s on EC2 `g4dn.12xlarge`

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

![Gif showing Agent integration](/assets/agent.gif)
<br/>

<!-- hatch-fancy-pypi-readme meta start -->

## 🚀 Deploying to Production

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
   This generates a OCI-compatible docker image that can be deployed anywhere docker runs.
   For best scalability and reliability of your LLM service in production, we recommend deploy
   with BentoCloud。


### ☁️ BentoCloud

Deploy OpenLLM with [BentoCloud](https://www.bentoml.com/bento-cloud/), the
the serverless cloud for shipping and scaling AI applications.

1. **Create a BentoCloud account:** [sign up here](https://bentoml.com/cloud)
   for early access

2. **Log into your BentoCloud account:**

   ```bash
   bentoml cloud login --api-token <your-api-token> --endpoint <bento-cloud-endpoint>
   ```

> **Note**: Replace `<your-api-token>` and `<bento-cloud-endpoint>` with your
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
[code](./src/openllm/utils/analytics.py).

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
