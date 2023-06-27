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

import types
import typing as t

import pytest

import openllm


if t.TYPE_CHECKING:
    from openllm.models.auto.factory import _BaseAutoLLMClass

_FRAMEWORK_MAPPING = {"flan_t5": "google/flan-t5-small", "opt": "facebook/opt-125m"}
_PROMPT_MAPPING = {
    "qa": "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?",
    "default": "What is the weather in SF?",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    models, fname = t.cast(types.ModuleType, metafunc.module).__name__.partition(".")[-1].split(".")[1:]

    if "tf" in fname:
        framework = "tf"
    elif "flax" in fname:
        framework = "flax"
    else:
        framework = "pt"

    llm, runner_kwargs = t.cast(
        "_BaseAutoLLMClass",
        openllm[framework],  # type: ignore
    ).for_model(models, model_id=_FRAMEWORK_MAPPING[models], return_runner_kwargs=True, ensure_available=True)
    llm.ensure_model_id_exists()
    if "runner" in metafunc.function.__name__:
        llm = llm.to_runner(**runner_kwargs)
        llm.init_local(quiet=True)

    if "qa" in metafunc.fixturenames:
        metafunc.parametrize("prompt,llm,qa", [(_PROMPT_MAPPING["qa"], llm, True)])
    else:
        metafunc.parametrize("prompt,llm", [(_PROMPT_MAPPING["default"], llm)])
