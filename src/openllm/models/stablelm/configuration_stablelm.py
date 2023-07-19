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


class StableLMConfig(openllm.LLMConfig):
    """StableLM-Base-Alpha is a suite of 3B and 7B parameter decoder-only language models.

    It is pre-trained on a diverse collection of English datasets with a sequence
    length of 4096 to push beyond the context window limitations of existing open-source language models.

    StableLM-Tuned-Alpha is a suite of 3B and 7B parameter decoder-only language models
    built on top of the StableLM-Base-Alpha models and further fine-tuned on various chat and
    instruction-following datasets.

    Refer to [StableLM-tuned's model card](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
    and [StableLM-base's model card](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)
    for more information.
    """

    __config__ = {
        "name_type": "lowercase",
        "url": "https://github.com/Stability-AI/StableLM",
        "architecture": "GPTNeoXForCausalLM",
        "default_id": "stabilityai/stablelm-tuned-alpha-3b",
        "model_ids": [
            "stabilityai/stablelm-tuned-alpha-3b",
            "stabilityai/stablelm-tuned-alpha-7b",
            "stabilityai/stablelm-base-alpha-3b",
            "stabilityai/stablelm-base-alpha-7b",
        ],
    }

    class GenerationConfig:
        temperature: float = 0.9
        max_new_tokens: int = 128
        top_k: int = 0
        top_p: float = 0.9


START_STABLELM_COMMAND_DOCSTRING = """\
Run a LLMServer for StableLM model.

\b
> See more information about StableLM at [stabilityai/stablelm-base-alpha-3b](https://huggingface.co/stabilityai/stablelm-base-alpha-3b)

\b
## Usage

Currently, StableLM only supports PyTorch. Make sure ``torch`` is available in your system.

\b
StableLM Runner will use stabilityai/stablelm-base-alpha-3b as the default model. To change to any other StableLM
saved pretrained, or a fine-tune StableLM, provide ``OPENLLM_STABLELM_MODEL_ID='stabilityai/stablelm-tuned-alpha-3b'``
or provide `--model-id` flag when running ``openllm start stablelm``:

\b
$ openllm start stablelm --model-id 'stabilityai/stablelm-tuned-alpha-3b'
"""

SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

DEFAULT_PROMPT_TEMPLATE = """{system_prompt}<|USER|>{instruction}<|ASSISTANT|>"""
