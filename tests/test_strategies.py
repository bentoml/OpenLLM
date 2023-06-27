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

import pytest


if t.TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import bentoml
from bentoml._internal.resource import get_resource
from openllm import _strategies as strategy
from openllm._strategies import CascadingResourceStrategy


class GPURunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "amd.com/gpu")


def unvalidated_get_resource(x: dict[str, t.Any], y: str):
    return get_resource(x, y, validate=False)


@pytest.mark.parametrize("gpu_type", ["nvidia.com/gpu", "amd.com/gpu"])
def test_cascade_strategy_worker_count(monkeypatch: MonkeyPatch, gpu_type: str):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: 2}, 1) == 2
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: 2}, 2) == 4
    assert pytest.raises(
        ValueError,
        CascadingResourceStrategy.get_worker_count,
        GPURunnable,
        {gpu_type: 0},
        1,
    )
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: [2, 7]}, 1) == 2
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: [2, 7]}, 2) == 4

    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: [2, 7]}, 0.5) == 1
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: [2, 7, 9]}, 0.5) == 2
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: [2, 7, 8, 9]}, 0.5) == 2
    assert CascadingResourceStrategy.get_worker_count(GPURunnable, {gpu_type: [2, 5, 7, 8, 9]}, 0.4) == 2


@pytest.mark.parametrize("gpu_type", ["nvidia.com/gpu", "amd.com/gpu"])
def test_cascade_strategy_worker_env(monkeypatch: MonkeyPatch, gpu_type: str):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)

    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 1, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 1, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7]}, 1, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"

    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 2, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 2, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "0"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 2, 2)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "1"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7]}, 2, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7]}, 2, 2)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7"

    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7]}, 0.5, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,7"

    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7, 8, 9]}, 0.5, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,7"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7, 8, 9]}, 0.5, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "8,9"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 7, 8, 9]}, 0.25, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,7,8,9"

    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 6, 7, 8, 9]}, 0.4, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "2,6"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 6, 7, 8, 9]}, 0.4, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "7,8"
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: [2, 6, 7, 8, 9]}, 0.4, 2)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "9"


@pytest.mark.parametrize("gpu_type", ["nvidia.com/gpu", "amd.com/gpu"])
def test_cascade_strategy_disabled_via_env(monkeypatch: MonkeyPatch, gpu_type: str):
    monkeypatch.setattr(strategy, "get_resource", unvalidated_get_resource)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 1, 0)
    assert envs.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    envs = CascadingResourceStrategy.get_worker_env(GPURunnable, {gpu_type: 2}, 1, 1)
    assert envs.get("CUDA_VISIBLE_DEVICES") == "-1"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
