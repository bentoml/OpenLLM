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

import pytest

import openllm


@pytest.fixture
def qa_prompt() -> str:
    return (
        "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
    )


@pytest.fixture
def flan_t5_id() -> str:
    return "google/flan-t5-small"


def test_small_flan(qa_prompt: str, flan_t5_id: str):
    llm = openllm.AutoLLM.for_model("flan-t5", model_id=flan_t5_id)
    generate = llm(qa_prompt)
    assert generate


def test_small_runner_flan(qa_prompt: str, flan_t5_id: str):
    llm = openllm.Runner("flan-t5", model_id=flan_t5_id, init_local=True)
    generate = llm(qa_prompt)
    assert generate
