from __future__ import annotations

import openllm


class GPTNeoXConfig(openllm.LLMConfig):
    """LLaMA model was proposed in [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.

    It is a collection of foundation language models ranging from 7B to 65B parameters.

    Note that all variants of LlaMA including fine-tuning, quantisation format are all supported with ``openllm.Llama``.

    Refer to [LlaMA's model card](https://huggingface.co/docs/transformers/main/model_doc/llama)
    for more information.
    """

    __config__ = {
        "name_type": "lowercase",
        "url": "https://github.com/facebookresearch/llama",
        "default_id": "eleutherai/gpt-neox-20b",
        "model_ids": ["eleutherai/gpt-neox-20b"],
    }

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
