from __future__ import annotations
import dataclasses
import logging
import os
import sys
import typing as t

# import openllm here for OPENLLMDEVDEBUG
import openllm
import transformers

# Make sure to have at least one GPU to run this script

openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

# On notebook, make sure to install the following
# ! pip install -U openllm[fine-tune] @ git+https://github.com/bentoml/OpenLLM.git

from datasets import load_dataset


if t.TYPE_CHECKING:
    from peft import PeftModel

DEFAULT_MODEL_ID = "facebook/opt-6.7b"


def load_trainer(
    model: PeftModel,
    tokenizer: transformers.GPT2TokenizerFast,
    dataset_dict: t.Any,
    training_args: TrainingArguments,
):
    return transformers.Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        args=dataclasses.replace(
            transformers.TrainingArguments(training_args.output_dir),
            **dataclasses.asdict(training_args),
        ),
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


model, tokenizer = openllm.AutoLLM.for_model(
    "opt",
    model_id=model_args.model_id,
    quantize="int8",
    ensure_available=True,
).prepare_for_training(
    adapter_type="lora",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# ft on english_quotes
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = load_trainer(model, tokenizer, data, training_args)
model.config.use_cache = False  # silence just for warning, reenable for inference later

trainer.train()

trainer.model.save_pretrained(os.path.join(training_args.output_dir, "lora"))
