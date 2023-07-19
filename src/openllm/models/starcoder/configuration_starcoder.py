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


class StarCoderConfig(openllm.LLMConfig):
    """The StarCoder models are 15.5B parameter models trained on 80+ programming languages from [The Stack (v1.2)](https://huggingface.co/datasets/bigcode/the-stack), with opt-out requests excluded.

    The model uses [Multi Query Attention](https://arxiv.org/abs/1911.02150),
    [a context window of 8192 tokens](https://arxiv.org/abs/2205.14135), and was trained using the
    [Fill-in-the-Middle](https://arxiv.org/abs/2207.14255) objective on 1 trillion tokens.

    Refer to [StarCoder's model card](https://huggingface.co/bigcode/starcoder) for more information.
    """

    __config__ = {
        "name_type": "lowercase",
        "requires_gpu": True,
        "url": "https://github.com/bigcode-project/starcoder",
        "architecture": "GPTBigCodeForCausalLM",
        "requirements": ["bitsandbytes"],
        "workers_per_resource": 0.5,
        "default_id": "bigcode/starcoder",
        "model_ids": ["bigcode/starcoder", "bigcode/starcoderbase"],
    }

    class GenerationConfig:
        temperature: float = 0.2
        max_new_tokens: int = 256
        min_new_tokens: int = 32
        top_k: float = 50
        top_p: float = 0.95
        pad_token_id: int = 49152
        repetition_penalty: float = 1.2


START_STARCODER_COMMAND_DOCSTRING = """\
Run a LLMServer for StarCoder model.

\b
> See more information about StarCoder at [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

\b
## Usage

Currently, StarCoder only supports PyTorch. Make sure ``torch`` is available in your system.

\b
StarCoder Runner will use bigcode/starcoder as the default model. To change to any other StarCoder
saved pretrained, or a fine-tune StarCoder, provide ``OPENLLM_STARCODER_MODEL_ID='bigcode/starcoder'``
or provide `--model-id` flag when running ``openllm start starcoder``:

\b
$ openllm start starcoder --model-id 'bigcode/starcoder'
"""

DEFAULT_PROMPT_TEMPLATE = """{instruction}"""
