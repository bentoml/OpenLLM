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

from pathlib import Path

_TARGET_FILE = Path(__file__).parent.parent / "src" / "openllm" / "models" / "__init__.py"


def main() -> int:
    lines = [
        f"from . import {p.name} as {p.name}"
        for p in _TARGET_FILE.parent.glob("*/")
        if p.name not in {"__pycache__", "__init__.py"}
    ]
    lines.sort()
    with _TARGET_FILE.open("w") as f:
        f.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
