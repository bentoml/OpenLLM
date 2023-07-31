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
class OPTConfig(openllm.LLMConfig):
    """OPT was first introduced in [Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) and first released in [metaseq's repository](https://github.com/facebookresearch/metaseq) on May 3rd 2022 by Meta AI.

    OPT was predominantly pretrained with English text, but a small amount of non-English data is still present
    within the training corpus via CommonCrawl. The model was pretrained using a causal language modeling (CLM)
    objective. OPT belongs to the same family of decoder-only models like GPT-3. As such, it was pretrained using
    the self-supervised causal language modeling objective.

    Refer to [OPT's HuggingFace page](https://huggingface.co/docs/transformers/model_doc/opt) for more information.
    """
    __config__ = {
        "name_type": "lowercase",
        "trust_remote_code": False,
        "url": "https://huggingface.co/docs/transformers/model_doc/opt",
        "default_id": "facebook/opt-1.3b",
        "architecture": "OPTForCausalLM",
        "model_ids": [
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-66b",
        ],
        "fine_tune_strategies": (
            {
                "adapter_type": "lora",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
            },
        ),
    }
    format_outputs: bool = openllm.LLMConfig.Field(False, description="""Whether to format the outputs. This can be used when num_return_sequences > 1.""")
    class GenerationConfig:
        top_k: int = 15
        temperature: float = 0.75
        max_new_tokens: int = 1024
        num_return_sequences: int = 1
START_OPT_COMMAND_DOCSTRING = """\
Run a LLMServer for OPT model.

\b
> See more information about falcon at [facebook/opt-66b](https://huggingface.co/facebook/opt-66b)

\b
## Usage

By default, this model will use the PyTorch model for inference. However, this model supports both Flax and Tensorflow.

\b
- To use Flax, set the environment variable ``OPENLLM_OPT_FRAMEWORK="flax"``

\b
- To use Tensorflow, set the environment variable ``OPENLLM_OPT_FRAMEWORK="tf"``

\b
OPT Runner will use facebook/opt-2.7b as the default model. To change to any other OPT
saved pretrained, or a fine-tune OPT, provide ``OPENLLM_OPT_MODEL_ID='facebook/opt-6.7b'``
or provide `--model-id` flag when running ``openllm start opt``:

\b
$ openllm start opt --model-id facebook/opt-6.7b
"""
DEFAULT_PROMPT_TEMPLATE = """{instruction}"""
