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


class ChatGLMConfig(openllm.LLMConfig):
    """ChatGLM is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM) framework.

    With the quantization technique, users can deploy locally on consumer-grade graphics cards
    (only 6GB of GPU memory is required at the INT4 quantization level).

    ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue.
    The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning,
    feedback bootstrap, and reinforcement learning wit human feedback.
    With only about 6.2 billion parameters, the model is able to generate answers that are in line
    with human preference.

    Refer to [ChatGLM's GitHub page](https://github.com/THUDM/ChatGLM-6B) for more information.
    """

    __config__ = {
        "name_type": "lowercase",
        "trust_remote_code": True,
        "timeout": 3600000,
        "requires_gpu": True,
        "url": "https://github.com/THUDM/ChatGLM-6B",
        "requirements": ["cpm-kernels", "sentencepiece"],
        "architecture": "ChatGLMForConditionalGeneration",
        "default_id": "thudm/chatglm-6b",
        "model_ids": [
            "thudm/chatglm-6b",
            "thudm/chatglm-6b-int8",
            "thudm/chatglm-6b-int4",
            "thudm/chatglm2-6b",
            "thudm/chatglm2-6b-int4",
        ],
    }

    retain_history: bool = openllm.LLMConfig.Field(
        False,
        description="""Whether to retain history given to the model.
        If set to True, then the model will retain given history.""",
    )

    use_half_precision: bool = openllm.LLMConfig.Field(True, description="Whether to use half precision for model.")

    class GenerationConfig:
        max_new_tokens: int = 2048
        num_beams: int = 1
        top_p: float = 0.7
        temperature: float = 0.95


START_CHATGLM_COMMAND_DOCSTRING = """\
Run a LLMServer for ChatGLM model.

\b
> See more information about ChatGLM at [THUDM/ChatGLM-6b](https://huggingface.co/thudm/chatglm-6b)

\b
## Usage

Currently, ChatGLM only supports PyTorch. Make sure ``torch`` is available in your system.

\b
ChatGLM Runner will use THUDM/ChatGLM-6b as the default model. To change to any other ChatGLM
saved pretrained, or a fine-tune ChatGLM, provide ``OPENLLM_CHATGLM_MODEL_ID='thudm/chatglm-6b-int8'``
or provide `--model-id` flag when running ``openllm start chatglm``:

\b
$ openllm start chatglm --model-id='thudm/chatglm-6b-int8'
"""

DEFAULT_PROMPT_TEMPLATE = """{instruction}"""
