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

import inflection
import tomlkit

import openllm

START_COMMENT = f"<!-- {os.path.basename(__file__)}: start -->\n"
END_COMMENT = f"<!-- {os.path.basename(__file__)}: stop -->\n"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    with open(os.path.join(ROOT, "pyproject.toml"), "r") as f:
        deps = tomlkit.parse(f.read()).value["project"]["optional-dependencies"]

    with open(os.path.join(ROOT, "README.md"), "r") as f:
        readme = f.readlines()

    start_index, stop_index = readme.index(START_COMMENT), readme.index(END_COMMENT)
    formatted: dict[t.Literal["Model", "CPU", "GPU", "Optional"], list[str]] = {
        "Model": [],
        "CPU": [],
        "GPU": [],
        "Optional": [],
    }
    max_name_len_div = 0
    max_install_len_div = 0
    does_not_need_custom_installation: list[str] = []
    for name, config in openllm.CONFIG_MAPPING.items():
        dashed = inflection.dasherize(name)
        model_name = f"[{dashed}]({config.__openllm_url__})"
        if len(model_name) > max_name_len_div:
            max_name_len_div = len(model_name)
        formatted["Model"].append(model_name)
        formatted["GPU"].append("✅")
        formatted["CPU"].append("✅" if not config.__openllm_requires_gpu__ else "❌")
        instruction = "👾 (not needed)"
        if dashed in deps:
            instruction = f"`pip install openllm[{dashed}]`"
        else:
            does_not_need_custom_installation.append(model_name)
        if len(instruction) > max_install_len_div:
            max_install_len_div = len(instruction)
        formatted["Optional"].append(instruction)

    meta = ["\n"]

    # NOTE: headers
    meta += f"| Model {' ' * (max_name_len_div - 6)} | CPU | GPU | Optional {' ' * (max_install_len_div - 8)}|\n"
    # NOTE: divs
    meta += f"| {'-' * max_name_len_div}" + " | --- | --- | " + f"{'-' * max_install_len_div} |\n"
    # NOTE: rows
    for links, cpu, gpu, custom_installation in t.cast("tuple[str, str, str, str]", zip(*formatted.values())):
        meta += (
            "| "
            + links
            + " " * (max_name_len_div - len(links))
            + f" | {cpu}  | {gpu}  | "
            + custom_installation
            + " "
            * (
                max_install_len_div
                - len(custom_installation)
                - (0 if links not in does_not_need_custom_installation else 1)
            )
            + " |\n"
        )
    meta += "\n"

    # NOTE: adding notes
    meta += """\
> NOTE: We respect users' system disk space. Hence, OpenLLM doesn't enforce to
> install dependencies to run all models. If one wishes to use any of the
> aforementioned models, make sure to install the optional dependencies
> mentioned above.

"""

    readme = readme[:start_index] + [START_COMMENT] + meta + [END_COMMENT] + readme[stop_index + 1 :]

    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.writelines(readme)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
