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
class GPTNeoXConfig(openllm.LLMConfig):
    """GPTNeoX is an autoregressive language model trained on the Pile, whose weights will be made freely and openly available to the public through a permissive license.

    It is, to the best of our knowledge, the largest dense autoregressive model
    that has publicly available weights at the time of submission. The training and evaluation code, as well as the model weights,
    can be found at https://github.com/EleutherAI/gpt-neox.

    GPTNeoX has been used to fine-tune on various models, such as Dolly, StableLM, and Pythia.

    Note that OpenLLM provides first-class support for all of the aforementioned model. Users can
    also use `openllm start gpt-neox` to run all of the GPTNeoX variant's model

    Refer to [GPTNeoX's model card](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
    for more information.
    """
    __config__ = {
        "model_name": "gpt_neox",
        "start_name": "gpt-neox",
        "requires_gpu": True,
        "architecture": "GPTNeoXForCausalLM",
        "url": "https://github.com/EleutherAI/gpt-neox",
        "default_id": "eleutherai/gpt-neox-20b",
        "model_ids": ["eleutherai/gpt-neox-20b"],
    }
    use_half_precision: bool = openllm.LLMConfig.Field(True, description="Whether to use half precision for model.")
    class GenerationConfig:
        temperature: float = 0.9
        max_new_tokens: int = 100
START_GPT_NEOX_COMMAND_DOCSTRING = """\
Run a LLMServer for GPTNeoX model.

\b
> See more information about GPTNeoX at [HuggingFace's model card](https://huggingface.co/docs/transformers/model_doc/gpt_neox)

\b
## Usage

Currently, GPTNeoX only supports PyTorch. Make sure ``torch`` is available in your system.

\b
GPTNeoX Runner will use EleutherAI/gpt-neox-20b as the default model. To change to any other GPTNeoX
saved pretrained, or a fine-tune GPTNeoX, provide ``OPENLLM_GPT_NEOX_MODEL_ID='stabilityai/stablelm-tuned-alpha-3b'``
or provide `--model-id` flag when running ``openllm start gpt-neox``:

\b
$ openllm start gpt-neox --model-id 'stabilityai/stablelm-tuned-alpha-3b'
"""
DEFAULT_PROMPT_TEMPLATE = """{instruction}"""
