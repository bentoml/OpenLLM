#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path

# currently we are assuming the indentatio level is 2 for comments
START_COMMENT = f'# {os.path.basename(__file__)}: start\n'
END_COMMENT = f'# {os.path.basename(__file__)}: stop\n'
START_SPECIAL_COMMENT = f'# {os.path.basename(__file__)}: special start\n'
END_SPECIAL_COMMENT = f'# {os.path.basename(__file__)}: special stop\n'
START_ATTRS_COMMENT = f'# {os.path.basename(__file__)}: attrs start\n'
END_ATTRS_COMMENT = f'# {os.path.basename(__file__)}: attrs stop\n'
# Stubs for auto class
START_AUTO_STUBS_COMMENT = f'# {os.path.basename(__file__)}: auto stubs start\n'
END_AUTO_STUBS_COMMENT = f'# {os.path.basename(__file__)}: auto stubs stop\n'

ROOT = Path(__file__).parent.parent
_TARGET_FILE = ROOT / 'openllm-core' / 'src' / 'openllm_core' / '_configuration.py'
_TARGET_AUTO_FILE = ROOT / 'openllm-core' / 'src' / 'openllm_core' / 'config' / 'configuration_auto.py'

sys.path.insert(0, (ROOT / 'openllm-core' / 'src').__fspath__())
from openllm_core._configuration import GenerationConfig, ModelSettings, SamplingParams
from openllm_core.config.configuration_auto import CONFIG_MAPPING_NAMES
from openllm_core.utils import codegen
from openllm_core.utils.peft import PeftType


def process_annotations(annotations: str) -> str:
  if 'NotRequired' in annotations:
    return annotations[len('NotRequired[') : -1]
  elif 'Required' in annotations:
    return annotations[len('Required[') : -1]
  else:
    return annotations


_value_docstring = {
  'default_id': """Return the default model to use when using 'openllm start <model_id>'.
        This could be one of the keys in 'self.model_ids' or custom users model.

        This field is required when defining under '__config__'.
        """,
  'model_ids': """A list of supported pretrained models tag for this given runnable.

        For example:
            For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                            "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]

        This field is required when defining under '__config__'.
        """,
  'architecture': """The model architecture that is supported by this LLM.

        Note that any model weights within this architecture generation can always be run and supported by this LLM.

        For example:
            For GPT-NeoX implementation, it is based on GptNeoXForCausalLM, which supports dolly-v2, stablelm:

            ```bash
            openllm start gpt-neox --model-id stabilityai/stablelm-tuned-alpha-3b
            ```""",
  'add_generation_prompt': """Whether to add generation prompt token for formatting chat templates. This arguments will be used for chat-based models.""",
  'backend': """List of supported backend for this given LLM class. Currently, we support "pt" and "vllm".""",
  'serialisation': """Default serialisation format for different models. Some will default to use the legacy 'bin'. """,
  'url': 'The resolved url for this LLMConfig.',
  'trust_remote_code': 'Whether to always trust remote code',
  'service_name': 'Generated service name for this LLMConfig. By default, it is "generated_{model_name}_service.py"',
  'requirements': 'The default PyPI requirements needed to run this given LLM. By default, we will depend on bentoml, torch, transformers.',
  'model_type': 'The model type for this given LLM. By default, it should be causal language modeling. Currently supported "causal_lm" or "seq2seq_lm"',
  'name_type': """The default name typed for this model. "dasherize" will convert the name to lowercase and
        replace spaces with dashes. "lowercase" will convert the name to lowercase. If this is not set, then both
        `model_name` and `start_name` must be specified.""",
  'model_name': 'The normalized version of __openllm_start_name__, determined by __openllm_name_type__',
  'start_name': 'Default name to be used with `openllm start`',
  'timeout': 'The default timeout to be set for this given LLM.',
  'workers_per_resource': """The number of workers per resource. This is used to determine the number of workers to use for this model.
        For example, if this is set to 0.5, then OpenLLM will use 1 worker per 2 resources. If this is set to 1, then
        OpenLLM will use 1 worker per resource. If this is set to 2, then OpenLLM will use 2 workers per resource.

        See StarCoder for more advanced usage. See
        https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy for more details.

        By default, it is set to 1.
        """,
  'fine_tune_strategies': 'The fine-tune strategies for this given LLM.',
}

_transformed = {'fine_tune_strategies': 't.Dict[AdapterType, FineTuneConfig]'}


def main() -> int:
  with _TARGET_FILE.open('r') as f:
    processed = f.readlines()

  start_idx, end_idx = processed.index(' ' * 2 + START_COMMENT), processed.index(' ' * 2 + END_COMMENT)
  start_stub_idx, end_stub_idx = (
    processed.index(' ' * 4 + START_SPECIAL_COMMENT),
    processed.index(' ' * 4 + END_SPECIAL_COMMENT),
  )
  start_attrs_idx, end_attrs_idx = (
    processed.index(' ' * 4 + START_ATTRS_COMMENT),
    processed.index(' ' * 4 + END_ATTRS_COMMENT),
  )

  # NOTE: inline stubs __config__ attrs representation
  special_attrs_lines: list[str] = []
  for keys, ForwardRef in codegen.get_annotations(ModelSettings).items():
    special_attrs_lines.append(
      f"{' ' * 4}{keys}:{_transformed.get(keys, process_annotations(ForwardRef.__forward_arg__))}\n"
    )
  # NOTE: inline stubs for _ConfigAttr type stubs
  config_attr_lines: list[str] = []
  for keys, ForwardRef in codegen.get_annotations(ModelSettings).items():
    config_attr_lines.extend(
      [
        ' ' * 4 + line
        for line in [
          f'__openllm_{keys}__:{_transformed.get(keys, process_annotations(ForwardRef.__forward_arg__))}=Field(None)\n',
          f"'''{_value_docstring[keys]}'''\n",
        ]
      ]
    )
  # NOTE: inline runtime __getitem__ overload process
  lines: list[str] = []
  lines.append(' ' * 2 + '# NOTE: ModelSettings arguments\n')
  for keys, ForwardRef in codegen.get_annotations(ModelSettings).items():
    lines.extend(
      [
        ' ' * 2 + line
        for line in [
          '@overload\n',
          f"def __getitem__(self,item:t.Literal['{keys}'])->{_transformed.get(keys, process_annotations(ForwardRef.__forward_arg__))}:...\n",
        ]
      ]
    )
  # special case variables: generation_class, extras, sampling_class
  lines.append(' ' * 2 + '# NOTE: generation_class, sampling_class and extras arguments\n')
  lines.extend(
    [
      ' ' * 2 + line
      for line in [
        '@overload\n',
        "def __getitem__(self,item:t.Literal['generation_class'])->t.Type[openllm_core.GenerationConfig]:...\n",
        '@overload\n',
        "def __getitem__(self,item:t.Literal['sampling_class'])->t.Type[openllm_core.SamplingParams]:...\n",
        '@overload\n',
        "def __getitem__(self,item:t.Literal['extras'])->t.Dict[str, t.Any]:...\n",
      ]
    ]
  )
  lines.append(' ' * 2 + '# NOTE: GenerationConfig arguments\n')
  generation_config_anns = codegen.get_annotations(GenerationConfig)
  for keys, type_pep563 in generation_config_anns.items():
    lines.extend(
      [
        ' ' * 2 + line
        for line in ['@overload\n', f"def __getitem__(self,item:t.Literal['{keys}'])->{type_pep563}:...\n"]
      ]
    )
  lines.append(' ' * 2 + '# NOTE: SamplingParams arguments\n')
  for keys, type_pep563 in codegen.get_annotations(SamplingParams).items():
    if keys not in generation_config_anns:
      lines.extend(
        [
          ' ' * 2 + line
          for line in ['@overload\n', f"def __getitem__(self,item:t.Literal['{keys}'])->{type_pep563}:...\n"]
        ]
      )
  lines.append(' ' * 2 + '# NOTE: PeftType arguments\n')
  for keys in PeftType._member_names_:
    lines.extend(
      [
        ' ' * 2 + line
        for line in [
          '@overload\n',
          f"def __getitem__(self,item:t.Literal['{keys.lower()}'])->t.Dict[str, t.Any]:...\n",
        ]
      ]
    )

  processed = (
    processed[:start_attrs_idx]
    + [' ' * 4 + START_ATTRS_COMMENT, *special_attrs_lines, ' ' * 4 + END_ATTRS_COMMENT]
    + processed[end_attrs_idx + 1 : start_stub_idx]
    + [' ' * 4 + START_SPECIAL_COMMENT, *config_attr_lines, ' ' * 4 + END_SPECIAL_COMMENT]
    + processed[end_stub_idx + 1 : start_idx]
    + [' ' * 2 + START_COMMENT, *lines, ' ' * 2 + END_COMMENT]
    + processed[end_idx + 1 :]
  )
  with _TARGET_FILE.open('w') as f:
    f.writelines(processed)

  with _TARGET_AUTO_FILE.open('r') as f:
    processed = f.readlines()

  start_auto_stubs_idx, end_auto_stubs_idx = (
    processed.index(' ' * 2 + START_AUTO_STUBS_COMMENT),
    processed.index(' ' * 2 + END_AUTO_STUBS_COMMENT),
  )
  lines = []
  for model, class_name in CONFIG_MAPPING_NAMES.items():
    lines.extend(
      [
        ' ' * 2 + line
        for line in [
          '@t.overload\n',
          '@classmethod\n',
          f"def for_model(cls,model_name:t.Literal['{model}'],**attrs:t.Any)->openllm_core.config.{class_name}:...\n",
        ]
      ]
    )
  processed = (
    processed[:start_auto_stubs_idx]
    + [' ' * 2 + START_AUTO_STUBS_COMMENT, *lines, ' ' * 2 + END_AUTO_STUBS_COMMENT]
    + processed[end_auto_stubs_idx + 1 :]
  )
  with _TARGET_AUTO_FILE.open('w') as f:
    f.writelines(processed)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
