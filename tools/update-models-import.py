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
import os
import typing as t
from pathlib import Path
import openllm

_TARGET_FILE = Path(__file__).parent.parent / "src" / "openllm" / "models" / "__init__.py"

def comment_generator(comment_type: str, action: t.Literal["start", "stop"] = "start", indentation: int = 0) -> str: return " " * indentation + f"# {os.path.basename(__file__)}: {action} {comment_type}\n"

START_MODULE_COMMENT, STOP_MODULE_COMMENT = comment_generator("module"), comment_generator("module", "stop")
START_TYPES_COMMENT, STOP_TYPES_COMMENT = comment_generator("types", indentation=4), comment_generator("types", "stop", indentation=4)

@openllm.utils.apply(lambda v: sorted([" " * 4 + _ for _ in v], key=lambda k: k.split()[-1]))
def create_stubs_import() -> list[str]: return [f"from . import {p.name} as {p.name}\n" for p in _TARGET_FILE.parent.glob("*/") if p.name not in {"__pycache__", "__init__.py", ".DS_Store"}]
def create_module_import() -> str: return f"_MODELS: set[str] = {{{', '.join(sorted([repr(p.name) for p in _TARGET_FILE.parent.glob('*/') if p.name not in ['__pycache__', '__init__.py', '.DS_Store']]))}}}\n"

def main() -> int:
    with _TARGET_FILE.open("r") as f: processed = f.readlines()
    stubs_lines, module_line = create_stubs_import(), create_module_import()

    start_module_idx, stop_module_idx = processed.index(START_MODULE_COMMENT), processed.index(STOP_MODULE_COMMENT)
    start_types_idx, stop_types_idex = processed.index(START_TYPES_COMMENT), processed.index(STOP_TYPES_COMMENT)
    processed = processed[:start_module_idx] + [START_MODULE_COMMENT, module_line, STOP_MODULE_COMMENT] + processed[stop_module_idx+1:start_types_idx] + [START_TYPES_COMMENT, *stubs_lines, STOP_TYPES_COMMENT] + processed[stop_types_idex+1:]
    with _TARGET_FILE.open("w") as f: f.writelines(processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
