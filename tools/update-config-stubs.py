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

import typing as t
import os

from pathlib import Path

import openllm
import importlib
from openllm._configuration import ModelSettings

# currently we are assuming the indentatio level is 4 for comments
START_COMMENT = f"# {os.path.basename(__file__)}: start\n"
END_COMMENT = f"# {os.path.basename(__file__)}: stop\n"

_TARGET_FILE = Path(__file__).parent.parent / "src" / "openllm" / "_configuration.py"

_imported = importlib.import_module(ModelSettings.__module__)


def process_annotations(annotations: str) -> str:
    if "NotRequired" in annotations:
        return annotations[len("NotRequired[") : -1]
    elif "Required" in annotations:
        return annotations[len("Required[") : -1]
    else:
        return annotations


def main() -> int:
    transformed = {"fine_tune_strategies": "t.Dict[AdapterType, FineTuneConfig]"}
    with _TARGET_FILE.open("r") as f:
        processed = f.readlines()

    start_idx, end_idx = processed.index(" " * 4 + START_COMMENT), processed.index(" " * 4 + END_COMMENT)

    # convention to use t.TYPE_CHECKING
    lines = [" " * 4 + "if t.TYPE_CHECKING:\n"]
    for keys, ForwardRef in openllm.utils.codegen.get_annotations(ModelSettings).items():
        lines.extend(
            list(
                map(
                    lambda line: " " * 8 + line,
                    [
                        "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                        f'def __getitem__(self, item: t.Literal["{keys}"] = ...) -> {transformed.get(keys, process_annotations(ForwardRef.__forward_arg__))}: ...\n',
                    ],
                )
            )
        )
    # special case variables: generation_class, extras
    lines.extend(
        list(
            map(
                lambda line: " " * 8 + line,
                [
                    "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                    'def __getitem__(self, item: t.Literal["generation_class"] = ...) -> t.Type[GenerationConfig]: ...\n',
                    "@overload\n" if "overload" in dir(_imported) else "@t.overload\n",
                    'def __getitem__(self, item: t.Literal["extras"] = ...) -> t.Dict[str, t.Any]: ...\n',
                ],
            )
        )
    )

    processed = (
        processed[:start_idx] + [" " * 4 + START_COMMENT] + lines + [" " * 4 + END_COMMENT] + processed[end_idx + 1 :]
    )
    with _TARGET_FILE.open("w") as f:
        f.writelines(processed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
