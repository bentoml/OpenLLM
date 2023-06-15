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

"""
Fine-tuning OPT playground.

This script demonstrate how one can easily fine tune compatible model 
with [LoRA](https://arxiv.org/abs/2106.09685)

It requires at least one GPU to be available, so make sure to have it.

python -m openllm.playground.ft_opt_lora --help
"""
from __future__ import annotations

import logging
import os
import typing as t

# import openllm here for OPENLLMDEVDEBUG
import openllm

logger = logging.getLogger(__name__)

openllm.utils.configure_logging()

if len(openllm.utils.gpu_count()) < 1:
    raise RuntimeError("This script can only be run with system that GPU is available.")

_deps = ["bitsandbytes", "datasets", "peft", "accelerate"]

if openllm.utils.DEBUG:
    logger.info("Installing dependencies to run this script: %s", _deps)

    if os.system(f"pip install -U {' '.join(_deps)}") != 0:
        raise SystemExit(1)

# We set it here first so that we only use 1 GPU. This needs to be run before immporting bitsandbytes
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
# disable welcome message
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from peft import PeftModel

DEFAULT_MODEL_ID = "facebook/opt-2.7b"


def print_trainable_parameters(model: torch.nn.Module):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        "trainable params: %s || all params: %s || trainable%: %s",
        trainable_params,
        all_param,
        100 * trainable_params / all_param,
    )


class CastOutputToFloat(nn.Sequential):
    def forward(self, input: t.Any):
        return super().forward(input).to(torch.float32)


def load_model(model_id: str, max_memory: str) -> tuple[PeftModel, transformers.GPT2TokenizerFast]:
    opt = openllm.AutoLLM.for_model(
        "opt",
        model_id=model_id,
        max_memory=max_memory,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.float16,
        ensure_available=True,
    )

    model, tokenizer = opt.model, opt.tokenizer

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    wrapped_peft = get_peft_model(model, lora_config)
    print_trainable_parameters(wrapped_peft)

    return wrapped_peft, tokenizer


def load_trainer(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.GPT2TokenizerFast,
    dataset_dict: DatasetDict,
    output_dir: str,
):
    return transformers.Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=20,
            learning_rate=3e-4,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model ID to fine-tune on.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "outputs"),
        help="Directory to store output lora weights",
    )

    args = parser.parse_args()

    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"

    model, tokenizer = load_model(args.model_id, max_memory=max_memory)

    dataset = "Abirate/english_quotes"
    data = load_dataset(dataset)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = load_trainer(model, tokenizer, data, args.output_dir)
    model.config.use_cache = False  # silence just for warning, reenable for inference later
    trainer.train()
