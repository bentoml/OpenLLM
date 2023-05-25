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


class StarCoderConfig(openllm.LLMConfig, name_type="lowercase", requires_gpu=True):
    """Configuration for the StarCoder model."""

    class GenerationConfig:
        temperature: float = 0.9
        max_new_tokens: int = 256
        top_p: float = 0.95
        pad_token_id: int = 49152
        repetition_penalty: float = 1.0


START_STARCODER_COMMAND_DOCSTRING = """\
Run a LLMServer for StarCoder model and variants.

\b
> See more information about StarCoder at [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

\b
## Usage

Currently, StarCoder only supports PyTorch. Make sure ``torch`` is available in your system.

\b
StarCoder Runner will use bigcode/starcoder as the default model. To change any to any other StarCoder
saved pretrained, or a fine-tune StarCoder, provide ``OPENLLM_STARCODER_PRETRAINED='bigcode/starcoder'``
"""

DEFAULT_PROMPT_TEMPLATE = """{instruction}"""
