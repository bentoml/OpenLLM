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


class ChatGLMConfig(openllm.LLMConfig, name_type="lowercase", trust_remote_code=True, default_timeout=3600000):
    """Configuration for the ChatGLM model."""

    retain_history: bool = True
    """Whether to retain history given to the model. If set to True, then the model will retain given history."""

    use_half_precision: bool = True
    """Whether to use half precision for model."""

    class GenerationConfig:
        max_length: int = 2048
        num_beams: int = 1
        top_p: float = 0.7
        temperature: float = 0.95


START_CHATGLM_COMMAND_DOCSTRING = """\
Run a LLMServer for ChatGLM model and variants.

\b
> See more information about ChatGLM at [THUDM/ChatGLM-6b](https://huggingface.co/thudm/chatglm-6b)

\b
## Usage

Currently, ChatGLM only supports PyTorch. Make sure ``torch`` is available in your system.

\b
ChatGLM Runner will use THUDM/ChatGLM-6b as the default model. To change any to any other ChatGLM
saved pretrained, or a fine-tune ChatGLM, provide ``OPENLLM_CHATGLM_PRETRAINED='thudm/chatglm-6b-int8'``
"""
