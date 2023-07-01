#!/usr/bin/env python3
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

from __future__ import annotations

import importlib
import os
from pathlib import Path

import openllm
from openllm._configuration import GenerationConfig
from openllm._configuration import ModelSettings
from openllm._configuration import PeftType


# currently we are assuming the indentatio level is 4 for comments
START_COMMENT = f"# {os.path.basename(__file__)}: start\n"
END_COMMENT = f"# {os.path.basename(__file__)}: stop\n"
START_SPECIAL_COMMENT = f"# {os.path.basename(__file__)}: special start\n"
END_SPECIAL_COMMENT = f"# {os.path.basename(__file__)}: special stop\n"

_TARGET_FILE = Path(__file__).parent.parent / "src" / "openllm" / "_configuration.py"

_imported = importlib.import_module(ModelSettings.__module__)


def process_annotations(annotations: str) -> str:
    if "NotRequired" in annotations:
        return annotations[len("NotRequired[") : -1]
    elif "Required" in annotations:
        return annotations[len("Required[") : -1]
    else:
        return annotations


_value_docstring = {
    "default_id": """Return the default model to use when using 'openllm start <model_id>'.
        This could be one of the keys in 'self.model_ids' or custom users model.

        This field is required when defining under '__config__'.
        """,
    "model_ids": """A list of supported pretrained models tag for this given runnable.

        For example:
            For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                                "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]

        This field is required when defining under '__config__'.
        """,
    "url": """The resolved url for this LLMConfig.""",
    "requires_gpu": """Determines if this model is only available on GPU. By default it supports GPU and fallback to CPU.""",
    "trust_remote_code": """Whether to always trust remote code""",
    "service_name": """Generated service name for this LLMConfig. By default, it is 'generated_{model_name}_service.py'""",
    "requirements": """The default PyPI requirements needed to run this given LLM. By default, we will depend on
        bentoml, torch, transformers.""",
    "bettertransformer": """Whether to use BetterTransformer for this given LLM. This depends per model
        architecture. By default, we will use BetterTransformer for T5 and StableLM models,
        and set to False for every other models.
        """,
    "model_type": """The model type for this given LLM. By default, it should be causal language modeling.
        Currently supported 'causal_lm' or 'seq2seq_lm'
        """,
    "runtime": """The runtime to use for this model. Possible values are `transformers` or `ggml`. See
        LlaMA for more information.""",
    "name_type": """The default name typed for this model. "dasherize" will convert the name to lowercase and
        replace spaces with dashes. "lowercase" will convert the name to lowercase.""",
    "model_name": """The normalized version of __openllm_start_name__, determined by __openllm_name_type__""",
    "start_name": """Default name to be used with `openllm start`""",
    "env": """A EnvVarMixin instance for this LLMConfig.""",
    "timeout": """The default timeout to be set for this given LLM.""",
    "workers_per_resource": """The number of workers per resource. This is used to determine the number of workers to use for this model.
        For example, if this is set to 0.5, then OpenLLM will use 1 worker per 2 resources. If this is set to 1, then
        OpenLLM will use 1 worker per resource. If this is set to 2, then OpenLLM will use 2 workers per resource.

        See StarCoder for more advanced usage. See
        https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy for more details.

        By default, it is set to 1.
        """,
    "fine_tune_strategies": """The fine-tune strategies for this given LLM.""",
    "generation_class": """The result generated GenerationConfig class for this LLMConfig. This will be used
        to create the generation_config argument that can be used throughout the lifecycle.
        This class will also be managed internally by OpenLLM.""",
}


def main() -> int:
    transformed = {"fine_tune_strategies": "t.Dict[AdapterType, FineTuneConfig]"}
    with _TARGET_FILE.open("r") as f:
        processed = f.readlines()

    start_idx, end_idx = processed.index(" " * 4 + START_COMMENT), processed.index(" " * 4 + END_COMMENT)
    start_stub_idx, end_stub_idx = processed.index(" " * 8 + START_SPECIAL_COMMENT), processed.index(
        " " * 8 + END_SPECIAL_COMMENT
    )

    # NOTE: inline stubs for _ConfigAttr type stubs
    config_attr_lines: list[str] = []
    for keys, ForwardRef in openllm.utils.codegen.get_annotations(ModelSettings).items():
        config_attr_lines.extend(
            [
                " " * 8 + line
                for line in [
                    f"__openllm_{keys}__: {transformed.get(keys, process_annotations(ForwardRef.__forward_arg__))} = Field(None)\n",
                    f'"""{_value_docstring[keys]}"""\n',
                ]
            ]
        )
    # special case for generation_class
    config_attr_lines.extend(
        [
            " " * 8 + line
            for line in [
                f"__openllm_generation_class__: type[GenerationConfig] = Field(None)\n",
                f'"""{_value_docstring["generation_class"]}"""\n',
            ]
        ]
    )

    # NOTE: inline runtime __getitem__ overload process
    lines: list[str] = []
    for keys, ForwardRef in openllm.utils.codegen.get_annotations(ModelSettings).items():
        lines.extend(
            [
                " " * 4 + line
                for line in [
                    "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                    f'def __getitem__(self, item: t.Literal["{keys}"] = ...) -> {transformed.get(keys, process_annotations(ForwardRef.__forward_arg__))}: ...\n',
                ]
            ]
        )
    # special case variables: generation_class, extras
    lines.extend(
        [
            " " * 4 + line
            for line in [
                "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                'def __getitem__(self, item: t.Literal["generation_class"] = ...) -> t.Type[GenerationConfig]: ...\n',
                "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                'def __getitem__(self, item: t.Literal["extras"] = ...) -> t.Dict[str, t.Any]: ...\n',
            ]
        ]
    )
    for keys, type_pep563 in openllm.utils.codegen.get_annotations(GenerationConfig).items():
        lines.extend(
            [
                " " * 4 + line
                for line in [
                    "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                    f'def __getitem__(self, item: t.Literal["{keys}"] = ...) -> {type_pep563}: ...\n',
                ]
            ]
        )

    for keys in PeftType._member_names_:
        lines.extend(
            [
                " " * 4 + line
                for line in [
                    "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                    f'def __getitem__(self, item: t.Literal["{keys.lower()}"] = ...) -> dict[str, t.Any]: ...\n',
                ]
            ]
        )

    processed = (
        processed[:start_stub_idx]
        + [" " * 8 + START_SPECIAL_COMMENT]
        + config_attr_lines
        + [" " * 8 + END_SPECIAL_COMMENT]
        + processed[end_stub_idx + 1 : start_idx]
        + [" " * 4 + START_COMMENT]
        + lines
        + [" " * 4 + END_COMMENT]
        + processed[end_idx + 1 :]
    )
    with _TARGET_FILE.open("w") as f:
        f.writelines(processed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
