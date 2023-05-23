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
"""The following includes OpenLLM configuration and excerpt from 
[instruct_pipeline.py](https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py)"""

from __future__ import annotations

import typing as t

import openllm

if t.TYPE_CHECKING:
    from openllm.types import LLMTokenizer


class DollyV2Config(openllm.LLMConfig):
    """Configuration for the dolly-v2 model."""

    return_full_text: bool = False

    class GenerationConfig:
        temperature: float = 0.9
        max_length: int = 100
        top_p: float = 0.92
        top_k: int = 0
        max_new_tokens: int = 256


START_DOLLY_V2_COMMAND_DOCSTRING = """\
Run a LLMServer for dolly-v2 model and variants.

\b
> See more information about dolly-v2 at [databricks/dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b)

\b
## Usage

Currently, dolly-v2 only supports PyTorch. Make sure ``torch`` is available in your system.

\b
Dolly-v2 Runner will use databricks/dolly-v2-3b as the default model. To change any to any other dolly-v2
saved pretrained, or a fine-tune dolly-v2, provide ``OPENLLM_DOLLY_V2_PRETRAINED='databricks/dolly-v2-7b'``
"""

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
DEFAULT_PROMPT_TEMPLATE = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: LLMTokenizer, key: str) -> int:
    """
    Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer: the tokenizer
        key: the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]
