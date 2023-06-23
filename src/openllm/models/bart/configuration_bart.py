from __future__ import annotations

import openllm

START_BART_COMMAND_DOCSTRING = """\
Run a LLMServer for BART model.

\b
> See more information about BART at [huggingface/transformers](https://huggingface.co/docs/transformers/model_doc/bart)

\b
## Usage

By default, this model will use the PyTorch model for inference. However, this model supports Tensorflow.

\b
- To use Tensorflow, set the environment variable ``OPENLLM_BART_FRAMEWORK="tf"``
\b
BART Runner will use facebook/bart-large-cnn as the default model. To change any to any other BART
saved pretrained, or a fine-tune BART, provide ``OPENLLM_BART_MODEL_ID='facebook/bart-large-cnn'``
or provide `--model-id` flag when running ``openllm start bart``:

"""

DEFAULT_PROMPT_TEMPLATE = """Summarize: {instruction}"""

class BartConfig(openllm.LLMConfig):
    """BART was released in the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)
    - it is a sequence-to-sequence model trained with denoising as pretraining objective.

    Refer to [BART's page](https://huggingface.co/docs/transformers/model_doc/bart) for more information.
    """

    __config__ = {
        "url": "https://huggingface.co/docs/transformers/model_doc/bart",
        "default_id": "facebook/bart-large-cnn",
        "model_ids": [
            "facebook/bart-large-cnn",
            "facebook/bart-large-xsum",
            "facebook/bart-large-mnli",
            "facebook/bart-large",
            "facebook/bart-base",
            "facebook/bart-large-cnn",
            "facebook/bart-large-xsum",
            "facebook/bart-large-mnli",
            "facebook/bart-large",
            "facebook/bart-base",
        ],
        "model_type": "seq2seq_lm",
    }

    class GenerationConfig:
        temperature: float = 0.5
        max_new_tokens: int = 128
        top_k: int = 50
        top_p: float = 0.9
        use_cache: bool = True
        early_stopping: bool = True