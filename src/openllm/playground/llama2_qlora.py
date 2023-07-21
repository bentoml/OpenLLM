from __future__ import annotations
import dataclasses
import logging
import os
import sys
import typing as t

# import openllm here for OPENLLMDEVDEBUG
import openllm
import torch
import transformers

if t.TYPE_CHECKING:
    import peft

# Make sure to have at least one GPU to run this script

openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

# On notebook, make sure to install the following
# ! pip install -U openllm[fine-tune] @ git+https://github.com/bentoml/OpenLLM.git

import bitsandbytes as bnb
from datasets import load_dataset
from random import randint
from itertools import chain
from functools import partial
from random import randrange


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# Change this to the local converted path if you don't have access to the meta-llama model
DEFAULT_MODEL_ID = "meta-llama/Llama-2-7b-hf"
DATASET_NAME = "databricks/databricks-dolly-15k"


def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt


# template dataset to add prompt to each sample
def template_dataset(sample, tokenizer):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result


def prepare_datasets(tokenizer, dataset_name=DATASET_NAME):
    # Load dataset from the hub
    dataset = load_dataset(dataset_name, split="train")

    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])

    # apply prompt template per sample
    dataset = dataset.map(partial(template_dataset, tokenizer=tokenizer), remove_columns=list(dataset.features))
    # print random sample
    print("Sample from dolly-v2 ds:", dataset[randint(0, len(dataset))]["text"])

    # tokenize and chunk dataset
    lm_dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    ).map(
        partial(chunk, chunk_length=2048),
        batched=True,
    )

    # Print total number of samples
    print(f"Total number of samples: {len(lm_dataset)}")
    return lm_dataset


@openllm.utils.requires_dependencies("peft", extra="fine-tune")
def prepare_for_int4_training(
    model_id: str, gradient_checkpointing: bool = True, bf16: bool = True
) -> tuple[peft.PeftModel, transformers.LlamaTokenizerFast]:
    from peft.tuners.lora import LoraLayer

    llm = openllm.AutoLLM.for_model(
        "llama",
        model_id=model_id,
        ensure_available=True,
        quantize="int4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        use_cache=not gradient_checkpointing,
        device_map="auto",
    )
    print("Model summary:", llm.model)

    # get lora target modules
    modules = find_all_linear_names(llm.model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    model, tokenizer = llm.prepare_for_training(adapter_type="lora", use_gradient_checkpointing=gradient_checkpointing)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer


@dataclasses.dataclass
class TrainingArguments:
    per_device_train_batch_size: int = dataclasses.field(default=1)
    gradient_checkpointing: bool = dataclasses.field(default=True)
    bf16: bool = dataclasses.field(default=torch.cuda.get_device_capability()[0] == 8)
    learning_rate: float = dataclasses.field(default=5e-5)
    num_train_epochs: int = dataclasses.field(default=3)
    logging_steps: int = dataclasses.field(default=1)
    report_to: str = dataclasses.field(default="none")
    output_dir: str = dataclasses.field(default=os.path.join(os.getcwd(), "outputs", "llama"))
    save_strategy: str = dataclasses.field(default="no")


@dataclasses.dataclass
class ModelArguments:
    model_id: str = dataclasses.field(default=DEFAULT_MODEL_ID)
    seed: int = dataclasses.field(default=42)
    merge_weights: bool = dataclasses.field(default=False)


parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, training_args = t.cast(
        t.Tuple[ModelArguments, TrainingArguments], parser.parse_args_into_dataclasses()
    )


# import the model first hand
openllm.import_model("llama", model_id=model_args.model_id)


def train_loop(model_args: ModelArguments, training_args: TrainingArguments):
    import peft

    transformers.set_seed(model_args.seed)

    model, tokenizer = prepare_for_int4_training(
        model_args.model_id,
        gradient_checkpointing=training_args.gradient_checkpointing,
        bf16=training_args.bf16,
    )
    datasets = prepare_datasets(tokenizer)

    trainer = transformers.Trainer(
        model=model,
        args=dataclasses.replace(
            transformers.TrainingArguments(training_args.output_dir), **dataclasses.asdict(training_args)
        ),
        train_dataset=datasets,
        data_collator=transformers.default_data_collator,
    )

    trainer.train()

    if model_args.merge_weights:
        # note that this will requires larger GPU as we will load the whole model into memory

        # merge adapter weights with base model and save
        # save int4 model
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)

        # gc mem
        del model, trainer
        torch.cuda.empty_cache()

        model = peft.AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        # merge lora with base weights and save
        model = model.merge_and_unload()
        model.save_pretrained(
            os.path.join(os.getcwd(), "outputs", "merged_llama_lora"), safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(os.path.join(training_args.output_dir, "lora"))


train_loop(model_args, training_args)
