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
import itertools
from syrupy.extensions.json import JSONSnapshotExtension

if t.TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion
    from openllm._types import LiteralRuntime


_FRAMEWORK_MAPPING = {"flan_t5": "google/flan-t5-small", "opt": "facebook/opt-125m"}
_PROMPT_MAPPING = {
    "qa": "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?",
    "default": "What is the weather in SF?",
}


def parametrise_local_llm(
    model: str,
) -> t.Generator[tuple[openllm.LLMRunner | openllm.LLM[t.Any, t.Any], str], None, None]:
    if model not in _FRAMEWORK_MAPPING:
        pytest.skip(f"'{model}' is not yet supported in framework testing.")

    runtime_impl: tuple[LiteralRuntime, ...] = tuple()
    if model in openllm.MODEL_MAPPING_NAMES:
        runtime_impl += ("pt",)
    if model in openllm.MODEL_FLAX_MAPPING_NAMES:
        runtime_impl += ("flax",)
    if model in openllm.MODEL_TF_MAPPING_NAMES:
        runtime_impl += ("tf",)

    for framework, prompt in itertools.product(runtime_impl, _PROMPT_MAPPING.keys()):
        llm, runner_kwargs = openllm.infer_auto_class(framework).for_model(
            model, model_id=_FRAMEWORK_MAPPING[model], ensure_available=True, return_runner_kwargs=True
        )
        yield llm, prompt
        runner = llm.to_runner(**runner_kwargs)
        runner.init_local(quiet=True)
        yield runner, prompt


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    model = t.cast("t.LiteralString", t.cast(types.ModuleType, metafunc.module).__name__.split(".")[-1].strip("test_"))
    if "prompt" in metafunc.fixturenames and "llm" in metafunc.fixturenames:
        metafunc.parametrize("prompt,llm", [(p, llm) for p, llm in parametrise_local_llm(model)])


class ResponseComparator(JSONSnapshotExtension):
    ...


@pytest.fixture(scope="module", name="response_snapshot")
def fixture_response_snapshot(snapshot: SnapshotAssertion):
    snapshot.use_extension(ResponseComparator)
