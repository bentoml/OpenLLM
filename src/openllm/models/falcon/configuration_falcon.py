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


class FalconConfig(openllm.LLMConfig):
    """Falcon-7B is a 7B parameters causal decoder-only model built by TII and trained on 1,500B tokens of [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) enhanced with curated corpora.

    It is made available under the TII Falcon LLM License.

    Refer to [Falcon's HuggingFace page](https://huggingface.co/tiiuae/falcon-7b) for more information.
    """

    __config__ = {
        "name_type": "lowercase",
        "trust_remote_code": True,
        "requires_gpu": True,
        "timeout": int(36e6),
        "url": "https://falconllm.tii.ae/",
        "requirements": ["einops", "xformers"],
        "architecture": "FalconForCausalLM",
        "default_id": "tiiuae/falcon-7b",
        "model_ids": [
            "tiiuae/falcon-7b",
            "tiiuae/falcon-40b",
            "tiiuae/falcon-7b-instruct",
            "tiiuae/falcon-40b-instruct",
        ],
        "fine_tune_strategies": (
            {
                "adapter_type": "lora",
                "r": 64,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            },
        ),
    }

    class GenerationConfig:
        max_new_tokens: int = 200
        top_k: int = 10
        num_return_sequences: int = 1
        num_beams: int = 4
        early_stopping: bool = True


START_FALCON_COMMAND_DOCSTRING = """\
Run a LLMServer for FalconLM model.

\b
> See more information about falcon at [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b)

\b
## Usage

Currently, FalconLM only supports PyTorch. Make sure ``torch`` is available in your system.

\b
FalconLM Runner will use tiiuae/falcon-7b as the default model. To change to any other FalconLM
saved pretrained, or a fine-tune FalconLM, provide ``OPENLLM_FALCON_MODEL_ID='tiiuae/falcon-7b-instruct'``
or provide `--model-id` flag when running ``openllm start falcon``:

\b
$ openllm start falcon --model-id tiiuae/falcon-7b-instruct
"""

DEFAULT_PROMPT_TEMPLATE = """{context}
{user_name}: {instruction}
{agent}:
"""
