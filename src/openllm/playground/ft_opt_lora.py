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

This script demonstrate how one can easily fine tune OPT
with [LoRA](https://arxiv.org/abs/2106.09685) and in int8 with bitsandbytes.

It is based on one of the Peft examples fine tuning script.
It requires at least one GPU to be available, so make sure to have it.

python -m openllm.playground.ft_opt_lora --help
"""
from __future__ import annotations

import logging
import os
import typing as t

# import openllm here for OPENLLMDEVDEBUG
import openllm


openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

if len(openllm.utils.gpu_count()) < 1:
    raise RuntimeError("This script can only be run with system that GPU is available.")

_deps = ["bitsandbytes", "datasets", "peft", "accelerate"]

if openllm.utils.DEBUG:
    logger.info("Installing dependencies to run this script: %s", _deps)

    if os.system(f"pip install -U {' '.join(_deps)}") != 0:
        raise SystemExit(1)

from datasets import load_dataset
from peft import LoraConfig
from peft import get_peft_model


if openllm.utils.pkg.pkg_version_info("peft")[:2] >= (0, 4):
    from peft import prepare_model_for_kbit_training
else:
    from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training

import transformers


if t.TYPE_CHECKING:
    from peft import PeftModel

DEFAULT_MODEL_ID = "facebook/opt-6.7b"


def load_model(model_id: str) -> tuple[PeftModel, transformers.GPT2TokenizerFast]:
    opt = openllm.AutoLLM.for_model(
        "opt",
        model_id=model_id,
        load_in_8bit=True,
        ensure_available=True,
    )

    model, tokenizer = opt.model, opt.tokenizer

    # prep the model for int8 training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_trainer(
    model: PeftModel,
    tokenizer: transformers.GPT2TokenizerFast,
    dataset_dict: t.Any,
    output_dir: str,
):
    return transformers.Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=50,
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

    model, tokenizer = load_model(args.model_id)

    # ft on english_quotes
    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = load_trainer(model, tokenizer, data, args.output_dir)
    model.config.use_cache = False  # silence just for warning, reenable for inference later

    trainer.train()

    model.save_pretrained(os.path.join(args.output_dir, "lora"))
