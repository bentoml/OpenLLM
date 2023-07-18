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

import openllm


class LlaMAConfig(openllm.LLMConfig):
    """LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.

    It is a collection of foundation language models ranging from 7B to 65B parameters.

    Note that all variants of LlaMA including fine-tuning, quantisation format are all supported with ``openllm.Llama``.

    Refer to [LlaMA's model card](https://huggingface.co/docs/transformers/main/model_doc/llama)
    for more information.
    """

    __config__ = {
        "model_name": "llama",
        "start_name": "llama",
        "url": "https://github.com/facebookresearch/llama",
        "default_id": "decapoda-research/llama-7b-hf",
        "model_ids": [
            "decapoda-research/llama-65b-hf",
            "decapoda-research/llama-30b-hf",
            "decapoda-research/llama-13b-hf",
            "decapoda-research/llama-7b-hf-int8",
            "decapoda-research/llama-7b-hf",
            "openlm-research/open_llama_7b_v2",
            "openlm-research/open_llama_3b_v2",
            "openlm-research/open_llama_13b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_3b",
            "huggyllama/llama-65b",
            "huggyllama/llama-30b",
            "huggyllama/llama-13b",
            "huggyllama/llama-7b",
            "syzymon/long_llama_3b",  # NOTE: use ``openllm.LongLLaMA`` to load this variant. Otherwise it will be limited to context length of 2048
        ],
    }

    class GenerationConfig:
        max_new_tokens: int = 32
        temperature: float = 0.8
        top_p: float = 0.95


START_LLAMA_COMMAND_DOCSTRING = """\
Run a LLMServer for LlaMA model.

\b
> See more information about LlaMA at [LlaMA's model card](https://huggingface.co/docs/transformers/main/model_doc/llama

\b
## Usage

Currently, LlaMA only supports PyTorch. Make sure ``torch`` is available in your system.

\b
LlaMA Runner will use decapoda-research/llama-7b-hf as the default model. To change to any other LlaMA
saved pretrained, or a fine-tune LlaMA, provide ``OPENLLM_LLAMA_MODEL_ID='openlm-research/open_llama_7b_v2'``
or provide `--model-id` flag when running ``openllm start llama``:

\b
$ openllm start llama --model-id 'openlm-research/open_llama_7b_v2'
"""


DEFAULT_PROMPT_TEMPLATE = """{instruction}"""
