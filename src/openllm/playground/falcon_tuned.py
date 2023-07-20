from __future__ import annotations
import dataclasses
import logging
import os
import sys
import typing as t

import torch

# import openllm here for OPENLLMDEVDEBUG
import openllm
import transformers


# Make sure to have at least one GPU to run this script

openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

# On notebook, make sure to install the following
# ! pip install -U openllm[fine-tune] @ git+https://github.com/bentoml/OpenLLM.git


from datasets import load_dataset
from trl import SFTTrainer


DEFAULT_MODEL_ID = "ybelkada/falcon-7b-sharded-bf16"
DATASET_NAME = "timdettmers/openassistant-guanaco"


@dataclasses.dataclass
class TrainingArguments:
    per_device_train_batch_size: int = dataclasses.field(default=4)
    gradient_accumulation_steps: int = dataclasses.field(default=4)
    optim: str = dataclasses.field(default="paged_adamw_32bit")
    save_steps: int = dataclasses.field(default=10)
    warmup_steps: int = dataclasses.field(default=10)
    max_steps: int = dataclasses.field(default=500)
    logging_steps: int = dataclasses.field(default=10)
    learning_rate: float = dataclasses.field(default=2e-4)
    max_grad_norm: float = dataclasses.field(default=0.3)
    warmup_ratio: float = dataclasses.field(default=0.03)
    fp16: bool = dataclasses.field(default=True)
    group_by_length: bool = dataclasses.field(default=True)
    lr_scheduler_type: str = dataclasses.field(default="constant")
    output_dir: str = dataclasses.field(default=os.path.join(os.getcwd(), "outputs", "falcon"))


@dataclasses.dataclass
class ModelArguments:
    model_id: str = dataclasses.field(default=DEFAULT_MODEL_ID)
    max_sequence_length: int = dataclasses.field(default=512)


parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, training_args = t.cast(
        t.Tuple[ModelArguments, TrainingArguments], parser.parse_args_into_dataclasses()
    )

model, tokenizer = openllm.AutoLLM.for_model(
    "falcon",
    model_id=model_args.model_id,
    quantize="int4",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    ensure_available=True,
).prepare_for_training(
    adapter_type="lora",
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
)
model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(DATASET_NAME, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=model_args.max_sequence_length,
    tokenizer=tokenizer,
    args=dataclasses.replace(
        transformers.TrainingArguments(training_args.output_dir),
        **dataclasses.asdict(training_args),
    ),
)

# upcast layernorm in float32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()

trainer.model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
