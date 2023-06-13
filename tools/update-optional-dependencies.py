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
import shutil

import inflection
import tomlkit

import openllm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINE_TUNE_DEPS = ["peft", "bitsandbytes", "datasets", "accelerate"]
FLAN_T5_DEPS = ["flax", "jax", "jaxlib", "tensorflow", "keras"]
OPENAI_DEPS = ["openai", "tiktoken"]

_base_requirements = {
    inflection.dasherize(name): config.__openllm_requirements__
    for name, config in openllm.CONFIG_MAPPING.items()
    if config.__openllm_requirements__
}

# NOTE: update this table when adding new external dependencies
_deps_table = {
    "fine-tune": FINE_TUNE_DEPS,
    "flan-t5": FLAN_T5_DEPS,
    "openai": OPENAI_DEPS,
}

_base_requirements.update(_deps_table)


def main() -> int:
    with open(os.path.join(ROOT, "pyproject.toml"), "r") as f:
        pyproject = tomlkit.parse(f.read())

    table = tomlkit.table()

    for name, config in _base_requirements.items():
        table.add(name, config)

    table.add("all", [f"openllm[{k}]" for k in table.keys()])

    pyproject["project"]["optional-dependencies"] = table
    with open(os.path.join(ROOT, "pyproject.toml"), "w") as f:
        f.write(tomlkit.dumps(pyproject))

    if shutil.which("taplo"):
        return os.system(f"taplo fmt {os.path.join(ROOT, 'pyproject.toml')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
