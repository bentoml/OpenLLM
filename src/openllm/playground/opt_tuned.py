from __future__ import annotations

import dataclasses
import logging
import os
import sys
import typing as t

# import openllm here for OPENLLMDEVDEBUG
import openllm

openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

if len(openllm.utils.gpu_count()) < 1:
    raise RuntimeError("This script can only be run with system that GPU is available.")

_deps = ['"openllm[fine-tune]"']

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
        lora_alphadata=32,
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
    training_args: transformers.TrainingArguments,
):
    return transformers.Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


@dataclasses.dataclass
class TrainingArguments:
    per_device_train_batch_size: int = dataclasses.field(default=4)
    gradient_accumulation_steps: int = dataclasses.field(default=4)
    warmup_steps: int = dataclasses.field(default=10)
    max_steps: int = dataclasses.field(default=50)
    learning_rate: float = dataclasses.field(default=3e-4)
    fp16: bool = dataclasses.field(default=True)
    logging_steps: int = dataclasses.field(default=1)
    output_dir: str = dataclasses.field(default=os.path.join(os.getcwd(), "outputs", "opt"))


@dataclasses.dataclass
class ModelArguments:
    model_id: str = dataclasses.field(default=DEFAULT_MODEL_ID)


parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, training_args = t.cast(
        t.Tuple[ModelArguments, TrainingArguments], parser.parse_args_into_dataclasses()
    )


model, tokenizer = load_model(model_args.model_id)

# ft on english_quotes
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = load_trainer(model, tokenizer, data, training_args)
model.config.use_cache = False  # silence just for warning, reenable for inference later

trainer.train()

model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
