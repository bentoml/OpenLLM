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
    formatted: dict[t.Literal["Model", "CPU", "GPU", "URL", "Installation", "Model Ids"], list[str | list[str]]] = {
        "Model": [],
        "URL": [],
        "CPU": [],
        "GPU": [],
        "Installation": [],
        "Model Ids": [],
    }
    max_install_len_div = 0
    for name, config_cls in openllm.CONFIG_MAPPING.items():
        dashed = inflection.dasherize(name)
        formatted["Model"].append(dashed)
        formatted["URL"].append(config_cls.__openllm_url__)
        formatted["GPU"].append("✅")
        formatted["CPU"].append("✅" if not config_cls.__openllm_requires_gpu__ else "❌")
        formatted["Model Ids"].append(config_cls.__openllm_model_ids__)
        if dashed in deps:
            instruction = f'```bash\npip install "openllm[{dashed}]"\n```'
        else:
            instruction = "```bash\npip install openllm\n```"
        if len(instruction) > max_install_len_div:
            max_install_len_div = len(instruction)
        formatted["Installation"].append(instruction)

    meta: list[str] = ["\n", "<table align='center'>\n"]

    # NOTE: headers
    meta += ["<tr>\n"]
    meta.extend([f"<th>{header}</th>\n" for header in formatted.keys() if header not in ("URL",)])
    meta += ["</tr>\n"]
    # NOTE: rows
    for name, url, cpu, gpu, installation, model_ids in t.cast(
        t.Iterable[t.Tuple[str, str, str, str, str, t.List[str]]], zip(*formatted.values())
    ):
        meta += "<tr>\n"
        meta.extend(
            [
                f"\n<td><a href={url}>{name}</a></td>\n",
                f"<td>{cpu}</td>\n",
                f"<td>{gpu}</td>\n",
                f"<td>\n\n{installation}\n\n</td>\n",
            ]
        )
        format_with_links: list[str] = []
        for lid in model_ids:
            format_with_links.append(f"<li><a href=https://huggingface.co/{lid}><code>{lid}</code></a></li>")
        meta.append("<td>\n\n<ul>" + "\n".join(format_with_links) + "</ul>\n\n</td>\n")
        meta += "</tr>\n"
    meta.extend(["</table>\n", "\n"])

    readme = readme[:start_index] + [START_COMMENT] + meta + [END_COMMENT] + readme[stop_index + 1 :]

    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.writelines(readme)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
