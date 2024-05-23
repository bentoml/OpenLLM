from __future__ import annotations

import typing as t
from openllm_core._typing_compat import TypedDict
from datasets import load_dataset

if t.TYPE_CHECKING:
  from transformers import PreTrainedTokenizerBase

FIXED_OUTPUT_LENGTH = 128


class DatasetEntry(TypedDict):
  human: str
  gpt: str


class SampledRequest(TypedDict):
  prompt: str
  prompt_length: int
  output_length: int


def prepare_sharegpt_request(
  num_requests: int, tokenizer: PreTrainedTokenizerBase, max_output_length: int | None = None
) -> list[SampledRequest]:
  def transform(examples) -> DatasetEntry:
    human, gpt = [], []
    for example in examples['conversations']:
      human.append(example[0]['value'])
      gpt.append(example[1]['value'])
    return {'human': human, 'gpt': gpt}

  def process(examples, tokenizer, max_output_length: t.Optional[int]):
    # Tokenize the 'human' and 'gpt' values in batches
    prompt_token_ids = tokenizer(examples['human']).input_ids
    completion_token_ids = tokenizer(examples['gpt']).input_ids

    # Create the transformed entries
    return {
      'prompt': examples['human'],
      'prompt_length': [len(ids) for ids in prompt_token_ids],
      'output_length': [
        len(ids) if max_output_length is None else FIXED_OUTPUT_LENGTH for ids in completion_token_ids
      ],
    }

  def filter_length(examples) -> list[bool]:
    result = []
    for prompt_length, output_length in zip(examples['prompt_length'], examples['output_length']):
      if prompt_length < 4 or output_length < 4:
        result.append(False)
      elif prompt_length > 1024 or prompt_length + output_length > 2048:
        result.append(False)
      else:
        result.append(True)
    return result

  return (
    (
      dataset := load_dataset(
        'anon8231489123/ShareGPT_Vicuna_unfiltered',
        data_files='ShareGPT_V3_unfiltered_cleaned_split.json',
        split='train',
      )
    )
    .filter(lambda example: len(example['conversations']) >= 2, num_proc=8)
    .map(transform, remove_columns=dataset.column_names, batched=True)
    .map(
      process,
      fn_kwargs={'tokenizer': tokenizer, 'max_output_length': max_output_length},
      remove_columns=['human', 'gpt'],
      batched=True,
    )
    .filter(filter_length, batched=True)
    .shuffle(seed=42)
    .to_list()[:num_requests]
  )
