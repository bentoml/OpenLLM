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

import dataclasses
import os
import typing as t

import inflection
import tomlkit

import openllm

if t.TYPE_CHECKING:
    from tomlkit.items import Array
    from tomlkit.items import Table


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass(frozen=True)
class Classifier:
    identifier: t.Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "status": "Development Status",
            "environment": "Environment",
            "license": "License",
            "topic": "Topic",
            "os": "Operating System",
            "audience": "Intended Audience",
            "typing": "Typing",
            "language": "Programming Language",
        }
    )

    joiner: str = " :: "

    @staticmethod
    def status() -> dict[int, str]:
        return {
            v: status
            for v, status in zip(
                range(1, 8),
                [
                    "1 - Planning",
                    "2 - Pre-Alpha",
                    "3 - Alpha",
                    "4 - Beta",
                    "5 - Production/Stable",
                    "6 - Mature",
                    "7 - Inactive",
                ],
            )
        }

    @staticmethod
    def apache() -> str:
        return Classifier.create_classifier("license", "OSI Approved", "Apache Software License")

    @staticmethod
    def create_classifier(identifier: str, *decls: t.Any) -> str:
        cls_ = Classifier()
        if identifier not in cls_.identifier:
            raise ValueError(f"{identifier} is not yet supported (supported alias: {Classifier.identifier})")
        return cls_.joiner.join([cls_.identifier[identifier], *decls])

    @staticmethod
    def create_python_classifier(
        implementation: list[str] | None = None, supported_version: list[str] | None = None
    ) -> list[str]:
        if supported_version is None:
            supported_version = ["3.8", "3.9", "3.10", "3.11", "3.12"]
        if implementation is None:
            implementation = ["CPython", "PyPy"]
        base = [
            Classifier.create_classifier("language", "Python"),
            Classifier.create_classifier("language", "Python", "3"),
        ]
        base.append(Classifier.create_classifier("language", "Python", "3", "Only"))
        base.extend([Classifier.create_classifier("language", "Python", version) for version in supported_version])
        base.extend(
            [Classifier.create_classifier("language", "Python", "Implementation", impl) for impl in implementation]
        )
        return base

    @staticmethod
    def create_status_classifier(level: int) -> str:
        return Classifier.create_classifier("status", Classifier.status()[level])


@dataclasses.dataclass(frozen=True)
class Dependencies:
    name: str
    git_repo_url: t.Optional[str] = None
    branch: t.Optional[str] = None
    extensions: t.Optional[t.List[str]] = None
    subdirectory: t.Optional[str] = None
    requires_gpu: bool = False
    lower_constraint: t.Optional[str] = None
    upper_constraint: t.Optional[str] = None
    platform: t.Optional[t.Tuple[t.Literal["Linux", "Windows", "Darwin"], t.Literal["eq", "ne"]]] = None

    def with_options(self, **kwargs: t.Any) -> Dependencies:
        return dataclasses.replace(self, **kwargs)

    @property
    def has_constraint(self) -> bool:
        return self.lower_constraint is not None or self.upper_constraint is not None

    @property
    def pypi_extensions(self) -> str:
        return "" if self.extensions is None else f"[{','.join(self.extensions)}]"

    @staticmethod
    def platform_restriction(platform: t.LiteralString, op: t.Literal["eq", "ne"] = "eq") -> str:
        return f'platform_system{"==" if op == "eq" else "!="}"{platform}"'

    def to_str(self) -> str:
        deps: list[str] = []
        if self.lower_constraint is not None and self.upper_constraint is not None:
            dep = f"{self.name}{self.pypi_extensions}>={self.lower_constraint},<{self.upper_constraint}"
        elif self.lower_constraint is not None:
            dep = f"{self.name}{self.pypi_extensions}>={self.lower_constraint}"
        elif self.upper_constraint is not None:
            dep = f"{self.name}{self.pypi_extensions}<{self.upper_constraint}"
        elif self.subdirectory is not None:
            dep = f"{self.name}{self.pypi_extensions} @ git+https://github.com/{self.git_repo_url}.git#subdirectory={self.subdirectory}"
        elif self.branch is not None:
            dep = f"{self.name}{self.pypi_extensions} @ git+https://github.com/{self.git_repo_url}.git@{self.branch}"
        else:
            dep = f"{self.name}{self.pypi_extensions}"

        deps.append(dep)

        if self.platform:
            deps.append(self.platform_restriction(*self.platform))

        return ";".join(deps)

    @classmethod
    def from_tuple(cls, *decls: t.Any) -> Dependencies:
        return cls(*decls)


_BENTOML_EXT = ["grpc", "io"]
_TRANSFORMERS_EXT = ["torch", "tokenizers", "accelerate"]

_BASE_DEPENDENCIES = [
    Dependencies(name="bentoml", extensions=_BENTOML_EXT, lower_constraint="1.0.25"),
    Dependencies(name="transformers", extensions=_TRANSFORMERS_EXT, lower_constraint="4.29.0"),
    Dependencies(name="safetensors"),
    Dependencies(name="optimum"),
    Dependencies(name="attrs", lower_constraint="23.1.0"),
    Dependencies(name="cattrs", lower_constraint="23.1.0"),
    Dependencies(name="orjson"),
    Dependencies(name="inflection"),
    Dependencies(name="tabulate", extensions=["widechars"], lower_constraint="0.9.0"),
    Dependencies(name="httpx"),
    Dependencies(name="click", lower_constraint="8.1.6"),
    Dependencies(name="typing_extensions"),
    Dependencies(name="GitPython"),
    Dependencies(name="cuda-python", platform=("Darwin", "ne")),
    Dependencies(name="bitsandbytes", upper_constraint="0.42"),  # 0.41  works with CUDA 11.8
]

_NIGHTLY_MAPPING: dict[str, Dependencies] = {
    "bentoml": Dependencies.from_tuple("bentoml", "bentoml/bentoml", "main", _BENTOML_EXT),
    "peft": Dependencies.from_tuple("peft", "huggingface/peft", "main", None),
    "transformers": Dependencies.from_tuple("transformers", "huggingface/transformers", "main", _TRANSFORMERS_EXT),
    "optimum": Dependencies.from_tuple("optimum", "huggingface/optimum", "main", None),
    "accelerate": Dependencies.from_tuple("accelerate", "huggingface/accelerate", "main", None),
    "bitsandbytes": Dependencies.from_tuple("bitsandbytes", "TimDettmers/bitsandbytes", "main", None),
    "trl": Dependencies.from_tuple("trl", "lvwerra/trl", "main", None),
    "vllm": Dependencies.from_tuple("vllm", "vllm-project/vllm", "main", None, None, True, None),
}

_ALL_RUNTIME_DEPS = ["flax", "jax", "jaxlib", "tensorflow", "keras"]
FINE_TUNE_DEPS = ["peft>=0.4.0", "bitsandbytes", "datasets", "accelerate", "trl"]
FLAN_T5_DEPS = _ALL_RUNTIME_DEPS
OPT_DEPS = _ALL_RUNTIME_DEPS
OPENAI_DEPS = ["openai", "tiktoken"]
AGENTS_DEPS = ["transformers[agents]>=4.30", "diffusers", "soundfile"]
PLAYGROUND_DEPS = ["jupyter", "notebook", "ipython", "jupytext", "nbformat"]
GGML_DEPS = ["ctransformers"]
GPTQ_DEPS = ["auto-gptq[triton]"]
VLLM_DEPS = ["vllm", "ray"]

_base_requirements: dict[str, t.Any]= {
    inflection.dasherize(name): config_cls.__openllm_requirements__
    for name, config_cls in openllm.CONFIG_MAPPING.items()
    if config_cls.__openllm_requirements__
}

# shallow copy from locals()
_locals = locals().copy()

# NOTE: update this table when adding new external dependencies
# sync with openllm.utils.OPTIONAL_DEPENDENCIES
_base_requirements.update(
    {v: _locals.get(f"{inflection.underscore(v).upper()}_DEPS") for v in openllm.utils.OPTIONAL_DEPENDENCIES}
)

_base_requirements = {k: v for k, v in sorted(_base_requirements.items())}

fname = f"{os.path.basename(os.path.dirname(__file__))}/{os.path.basename(__file__)}"


def create_classifiers() -> Array:
    arr = tomlkit.array()
    arr.extend(
        [
            Classifier.create_status_classifier(5),
            Classifier.create_classifier("environment", "GPU", "NVIDIA CUDA"),
            Classifier.create_classifier("environment", "GPU", "NVIDIA CUDA", "12"),
            Classifier.create_classifier("environment", "GPU", "NVIDIA CUDA", "11.8"),
            Classifier.create_classifier("environment", "GPU", "NVIDIA CUDA", "11.7"),
            Classifier.apache(),
            Classifier.create_classifier("topic", "Scientific/Engineering", "Artificial Intelligence"),
            Classifier.create_classifier("topic", "Software Development", "Libraries"),
            Classifier.create_classifier("os", "OS Independent"),
            Classifier.create_classifier("audience", "Developers"),
            Classifier.create_classifier("audience", "Science/Research"),
            Classifier.create_classifier("audience", "System Administrators"),
            Classifier.create_classifier("typing", "Typed"),
            *Classifier.create_python_classifier(),
        ]
    )
    return arr.multiline(True)


def create_optional_table() -> Table:
    all_array = tomlkit.array()
    all_array.extend([f"openllm[{k}]" for k in _base_requirements])

    table = tomlkit.table(is_super_table=True)
    _base_requirements.update({"all": all_array.multiline(True)})
    table.update({k: v for k, v in sorted(_base_requirements.items())})
    table.add(tomlkit.nl())

    return table


def create_url_table() -> Table:
    table = tomlkit.table()
    _urls = {
        "Blog": "https://modelserving.com",
        "Chat": "https://discord.gg/openllm",
        "Documentation": "https://github.com/bentoml/openllm#readme",
        "GitHub": "https://github.com/bentoml/openllm",
        "History": "https://github.com/bentoml/openllm/blob/main/CHANGELOG.md",
        "Homepage": "https://bentoml.com",
        "Tracker": "https://github.com/bentoml/openllm/issues",
        "Twitter": "https://twitter.com/bentomlai",
    }
    table.update({k: v for k, v in sorted(_urls.items())})
    return table

def build_cli_extensions() -> Table:
    table = tomlkit.table()
    ext: dict[str, str] = {"openllm": "openllm.cli.entrypoint:cli"}
    ext.update({f"openllm-{inflection.dasherize(ke)}": f"openllm.cli.ext.{ke}:cli" for ke in sorted([fname[:-3]
                                                                                   for fname in os.listdir(os.path.abspath(os.path.join(ROOT, "src", "openllm", "cli", "ext")))
                                                                                   if fname.endswith(".py") and not fname.startswith("__")])})
    table.update(ext)
    return table

def main() -> int:
    with open(os.path.join(ROOT, "pyproject.toml"), "r") as f:
        pyproject = tomlkit.parse(f.read())

    dependencies_array = tomlkit.array()
    dependencies_array.extend([v.to_str() for v in _BASE_DEPENDENCIES])

    pyproject["project"]["urls"] = create_url_table()
    pyproject["project"]["scripts"] = build_cli_extensions()
    pyproject["project"]["classifiers"] = create_classifiers()
    pyproject["project"]["optional-dependencies"] = create_optional_table()
    pyproject["project"]["dependencies"] = dependencies_array.multiline(True)

    with open(os.path.join(ROOT, "pyproject.toml"), "w") as f:
        f.write(tomlkit.dumps(pyproject))

    with open(os.path.join(ROOT, "nightly-requirements.txt"), "w") as f:
        f.write(f"# This file is generated by `{fname}`. DO NOT EDIT\n-e .[playground,flan-t5]\n")
        f.writelines([f"{v.to_str()}\n" for v in _NIGHTLY_MAPPING.values() if not v.requires_gpu])
    with open(os.path.join(ROOT, "nightly-requirements-gpu.txt"), "w") as f:
        f.write(f"# This file is generated by `{fname}`. # DO NOT EDIT\n")
        f.write(
            "# For Jax, Flax, Tensorflow, PyTorch CUDA support, please refers to their official installation for your specific setup.\n"
        )
        f.write("-r nightly-requirements.txt\n-e .[all]\n")
        f.writelines([f"{v.to_str()}\n" for v in _NIGHTLY_MAPPING.values() if v.requires_gpu])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
