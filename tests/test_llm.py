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
import openllm
import pytest
from openllm._llm import make_tag

HF_INTERNAL_T5_TESTING = "hf-internal-testing/tiny-random-t5"


def test_tag_generation():
    tag = make_tag(HF_INTERNAL_T5_TESTING)
    assert tag.version == "2f582cd79ed5795b71539951d237945bc1c5ac7e"


def patch_hash_from_file(_: str, algorithm: t.LiteralString = "sha1") -> str:
    return "d88a1a40e354a0c7fa6f9055938594e6a4c712e0"


def test_tag_generation_from_custom_path(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(openllm._llm, "generate_hash_from_file", patch_hash_from_file)
    local_path = tmp_path_factory.mktemp("local_t5")
    llm = openllm.AutoLLM.for_model("flan-t5", model_id=HF_INTERNAL_T5_TESTING)
    llm.save_pretrained(local_path)

    with caplog.at_level("WARNING"):
        tag = make_tag(local_path.resolve().__fspath__())

    assert tag.version == "d88a1a40e354a0c7fa6f9055938594e6a4c712e0"
    assert "Given 'model_id" in caplog.text


def test_tag_generation_quiet_log(tmp_path_factory: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture):
    local_path = tmp_path_factory.mktemp("local_t5")
    llm = openllm.AutoLLM.for_model("flan-t5", model_id=HF_INTERNAL_T5_TESTING)
    llm.save_pretrained(local_path)

    with caplog.at_level("WARNING"):
        make_tag(local_path.resolve().__fspath__(), quiet=True)
    assert not caplog.text


def test_tag_generation_debug_log(caplog: pytest.LogCaptureFixture):
    with caplog.at_level("DEBUG"):
        make_tag(HF_INTERNAL_T5_TESTING)
    assert (
        "The full tag to be saved under model store: 'pt-hf-internal-testing-tiny-random-t5:2f582cd79ed5795b71539951d237945bc1c5ac7e'"
        in caplog.text
    )
