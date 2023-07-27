# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import typing as t
import openllm
class LlaMAConfig(openllm.LLMConfig):
    """LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.

    It is a collection of foundation language models ranging from 7B to 65B parameters.

    LlaMA also include support for the recent propsed [LlaMA-2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

    Note that all variants of LlaMA including fine-tuning, quantisation format are all supported with ``openllm.LlaMA``.

    Refer to [LlaMA's model card](https://huggingface.co/docs/transformers/main/model_doc/llama)
    for more information.
    """
    use_llama2_prompt: bool = openllm.LLMConfig.Field(True, description="Whether to use the prompt format for LlaMA 2. Disable this when working with LlaMA 1.")
    __config__ = {
        "model_name": "llama",
        "start_name": "llama",
        "url": "https://github.com/facebookresearch/llama",
        "default_id": "huggyllama/llama-7b",
        "default_implementation": {"cpu": "pt", "nvidia.com/gpu": "pt"},
        "architecture": "LlamaForCausalLM",
        "requirements": ["fairscale", "sentencepiece"],
        "model_ids": [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-7b-hf",
            "NousResearch/llama-2-70b-chat-hf",
            "NousResearch/llama-2-13b-chat-hf",
            "NousResearch/llama-2-7b-chat-hf",
            "NousResearch/llama-2-70b-hf",
            "NousResearch/llama-2-13b-hf",
            "NousResearch/llama-2-7b-hf",
            "openlm-research/open_llama_7b_v2",
            "openlm-research/open_llama_3b_v2",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-65b",
            "huggyllama/llama-30b",
            "huggyllama/llama-13b",
            "huggyllama/llama-7b",
        ],
        "tokenizer_class": "LlamaTokenizerFast",
        "fine_tune_strategies": (
            {
                "adapter_type": "lora",
                "r": 64,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none",
            },
        ),
    }
    class GenerationConfig:
        max_new_tokens: int = 256
        temperature: float = 0.45
        top_p: float = 0.95
        top_k: int = 12
    class SamplingParams:
        best_of: int = 1
        presence_penalty: float = 0.5
START_LLAMA_COMMAND_DOCSTRING = """\
Run a LLMServer for LlaMA model.

\b
> See more information about LlaMA at [LlaMA's model card](https://huggingface.co/docs/transformers/main/model_doc/llama

\b
## Usage

By default, this model will use [vLLM](https://github.com/vllm-project/vllm) for inference.
This model will also supports PyTorch.

\b
- To use PyTorch, set the environment variable ``OPENLLM_LLAMA_FRAMEWORK="pt"``

\b
LlaMA Runner will use decapoda-research/llama-7b-hf as the default model. To change to any other LlaMA
saved pretrained, or a fine-tune LlaMA, provide ``OPENLLM_LLAMA_MODEL_ID='openlm-research/open_llama_7b_v2'``
or provide `--model-id` flag when running ``openllm start llama``:

\b
$ openllm start llama --model-id 'openlm-research/open_llama_7b_v2'

\b
OpenLLM also supports running LlaMA-2 and its fine-tune and variants. To import the LlaMA weights, one can use the following:

\b
$ CONVERTER=hf-llama2 openllm import llama /path/to/llama-2
"""
SYSTEM_MESSAGE = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
SINST_KEY, EINST_KEY, SYS_KEY, EOS_TOKEN, BOS_TOKEN = "[INST]", "[/INST]", "<<SYS>>", "</s>", "<s>"
# TODO: support history and v1 prompt implementation
_v1_prompt, _v2_prompt = """{instruction}""", """{start_key} {sys_key}\n{system_message}\n{sys_key}\n\n{instruction}\n{end_key} """.format(start_key=SINST_KEY, sys_key=SYS_KEY, system_message=SYSTEM_MESSAGE, instruction="{instruction}", end_key=EINST_KEY)
PROMPT_MAPPING = {"v1": _v1_prompt, "v2": _v2_prompt}
def _get_prompt(model_type: t.Literal["v1", "v2"]) -> str: return PROMPT_MAPPING[model_type]
DEFAULT_PROMPT_TEMPLATE = _get_prompt
