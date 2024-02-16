#!/usr/bin/env python3
from __future__ import annotations
import os, sys
from pathlib import Path

# currently we are assuming the indentatio level is 2 for comments
START_COMMENT = f'# {os.path.basename(__file__)}: start\n'
END_COMMENT = f'# {os.path.basename(__file__)}: stop\n'
# Stubs for auto class
START_AUTO_STUBS_COMMENT = f'# {os.path.basename(__file__)}: auto stubs start\n'
END_AUTO_STUBS_COMMENT = f'# {os.path.basename(__file__)}: auto stubs stop\n'
# Stubs for actual imports
START_IMPORT_STUBS_COMMENT = f'# {os.path.basename(__file__)}: import stubs start\n'
END_IMPORT_STUBS_COMMENT = f'# {os.path.basename(__file__)}: import stubs stop\n'

ROOT = Path(__file__).parent.parent
_TARGET_FILE = ROOT/'openllm-core'/'src'/'openllm_core'/'_configuration.py'
_TARGET_AUTO_FILE = ROOT/'openllm-core'/'src'/'openllm_core'/'config'/'configuration_auto.py'
_TARGET_CORE_INIT_FILE = ROOT/'openllm-core'/'src'/'openllm_core'/'config'/'__init__.py'
_TARGET_INIT_FILE = ROOT/'openllm-python'/'src'/'openllm'/'__init__.pyi'
_TARGET_IMPORT_UTILS_FILE = ROOT/'openllm-core'/'src'/'openllm_core'/'utils'/'import_utils.pyi'

sys.path.insert(0, (ROOT/'openllm-core'/'src').__fspath__())
from openllm_core._configuration import GenerationConfig, ModelSettings
from openllm_core.config.configuration_auto import CONFIG_MAPPING_NAMES
from openllm_core.utils import codegen, import_utils as iutils
# from openllm_core.utils.peft import PeftType


def process_annotations(annotations: str) -> str:
  if 'NotRequired' in annotations:
    return annotations[len('NotRequired['): -1]
  elif 'Required' in annotations:
    return annotations[len('Required['): -1]
  else:
    return annotations


_value_docstring = {
  'default_id': """Return the default model to use when using 'openllm start <model_id>'.
        This could be one of the keys in 'self.model_ids' or custom users model.

        This field is required when defining under 'metadata_config'.
        """,
  'model_ids': """A list of supported pretrained models tag for this given runnable.

        For example:
            For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                            "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]

        This field is required when defining under 'metadata_config'.
        """,
  'architecture': """The model architecture that is supported by this LLM.

        Note that any model weights within this architecture generation can always be run and supported by this LLM.

        For example:
            For GPT-NeoX implementation, it is based on GptNeoXForCausalLM, which supports dolly-v2, stablelm:

            ```bash
            openllm start stabilityai/stablelm-tuned-alpha-3b
            ```""",
  'add_generation_prompt': """Whether to add generation prompt token for formatting chat templates. This arguments will be used for chat-based models.""",
  'backend': """List of supported backend for this given LLM class. Currently, we support `pt` and `vllm`.""",
  'serialisation': """Default serialisation format for different models. Some will default to use the legacy `bin`. """,
  'url': 'The resolved url for this LLMConfig.',
  'trust_remote_code': 'Whether to always trust remote code',
  'service_name': 'Generated service name for this LLMConfig. By default, it is `generated_{model_name}_service.py`',
  'requirements': 'The default PyPI requirements needed to run this given LLM. By default, we will depend on bentoml, torch, transformers.',
  'model_type': 'The model type for this given LLM. By default, it should be causal language modeling. Currently supported `causal_lm` or `seq2seq_lm`',
  'name_type': """The default name typed for this model. "dasherize" will convert the name to lowercase and
        replace spaces with dashes. `lowercase` will convert the name to lowercase. If this is not set, then both
        `model_name` and `start_name` must be specified.""",
  'model_name': 'The normalized version of __openllm_start_name__, determined by __openllm_name_type__',
  'start_name': 'Default name to be used with `openllm start`',
  'timeout': 'The default timeout to be set for this given LLM.',
  'fine_tune_strategies': 'The fine-tune strategies for this given LLM.',
}

_transformed = {'fine_tune_strategies': 't.Dict[AdapterType, FineTuneConfig]'}


def main() -> int:
  with _TARGET_FILE.open('r') as f:
    processed = f.readlines()

  start_idx, end_idx = processed.index(' ' * 2 + START_COMMENT), processed.index(' ' * 2 + END_COMMENT)

  # NOTE: inline runtime __getitem__ overload process
  lines: list[str] = []
  lines.append(' ' * 2 + '# NOTE: ModelSettings arguments\n')
  for keys, ForwardRef in codegen.get_annotations(ModelSettings).items():
    lines.extend(
      [
        ' ' * 2 + line
        for line in [
          '@overload\n',
          f"def __getitem__(self, item: t.Literal['{keys}']) -> {process_annotations(ForwardRef.__forward_arg__)}: ...\n",
        ]
      ]
    )
  lines.append(' ' * 2 + '# NOTE: GenerationConfig arguments\n')
  generation_config_anns = codegen.get_annotations(GenerationConfig)
  for keys, type_pep563 in generation_config_anns.items():
    lines.extend(
      [
        ' ' * 2 + line
        for line in ['@overload\n', f"def __getitem__(self, item: t.Literal['{keys}']) -> {type_pep563}: ...\n"]
      ]
    )

  processed = processed[:start_idx] + [' ' * 2 + START_COMMENT, *lines, ' ' * 2 + END_COMMENT] + processed[end_idx + 1:]
  with _TARGET_FILE.open('w') as f: f.writelines(processed)

  with _TARGET_AUTO_FILE.open('r') as f: processed = f.readlines()

  start_auto_stubs_idx, end_auto_stubs_idx = processed.index(' ' * 2 + START_AUTO_STUBS_COMMENT), processed.index(' ' * 2 + END_AUTO_STUBS_COMMENT)
  lines = []
  for model, class_name in CONFIG_MAPPING_NAMES.items():
    lines.extend(
      [
        ' ' * 2 + line
        for line in [
          '@t.overload\n',
          '@classmethod\n',
          f"def for_model(cls, model_name: t.Literal['{model}'], **attrs: t.Any) -> openllm_core.config.{class_name}: ...\n",
        ]
      ]
    )
  processed = processed[:start_auto_stubs_idx] + [' ' * 2 + START_AUTO_STUBS_COMMENT, *lines, ' ' * 2 + END_AUTO_STUBS_COMMENT] + processed[end_auto_stubs_idx + 1:]
  with _TARGET_AUTO_FILE.open('w') as f: f.writelines(processed)

  with _TARGET_INIT_FILE.open('r') as f: processed = f.readlines()

  start_import_stubs_idx, end_import_stubs_idx = processed.index(START_IMPORT_STUBS_COMMENT), processed.index(END_IMPORT_STUBS_COMMENT)
  mm = {
    '_configuration': ("GenerationConfig", "LLMConfig"),
    '_schemas': ('GenerationInput', 'GenerationOutput', 'MetadataOutput', 'MessageParam'),
    'utils': ('api',),
  }
  lines = [
    'from openllm_client import AsyncHTTPClient as AsyncHTTPClient, HTTPClient as HTTPClient',
    f'from openlm_core.config import CONFIG_MAPPING as CONFIG_MAPPING, CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES, AutoConfig as AutoConfig, {", ".join([a+" as "+a for a in CONFIG_MAPPING_NAMES.values()])}',
  ]
  lines.extend([f'from openllm_core.{module} import {", ".join([a+" as "+a for a in attr])}' for module, attr in mm.items()])
  processed = processed[:start_import_stubs_idx] + [START_IMPORT_STUBS_COMMENT, '\n'.join(lines) + '\n', END_IMPORT_STUBS_COMMENT] + processed[end_import_stubs_idx + 1:]
  with _TARGET_INIT_FILE.open('w') as f: f.writelines(processed)

  lines = [
    '# fmt: off\n',
    f'# AUTOGENERATED BY {os.path.basename(__file__)}. DO NOT EDIT\n',
    'from .configuration_auto import CONFIG_MAPPING as CONFIG_MAPPING, CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES, AutoConfig as AutoConfig\n',
    *[f'from .configuration_{k} import {a} as {a}\n' for k, a in CONFIG_MAPPING_NAMES.items()]
  ]
  with _TARGET_CORE_INIT_FILE.open('w') as f: f.writelines(lines)

  lines = [
    '# fmt: off\n',
    f'# AUTOGENERATED BY {os.path.basename(__file__)}. DO NOT EDIT\n', 'import typing as t\n',
    'def is_autoawq_available() -> bool: ...\n', 'def is_vllm_available() -> bool: ...\n',
    *[f'def {k}() -> bool: ...\n' for k in iutils.caller],
    'ENV_VARS_TRUE_VALUES: t.Set[str] = ...\n', 'OPTIONAL_DEPENDENCIES: t.Set[str] = ...\n',
  ]
  with _TARGET_IMPORT_UTILS_FILE.open('w') as f: f.writelines(lines)

  return 0


if __name__ == '__main__': raise SystemExit(main())
