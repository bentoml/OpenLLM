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
import sys
import typing as t

import pytest


if t.TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import bentoml
from bentoml._internal.resource import get_resource
from openllm import _strategies as strategy
from openllm._strategies import CascadingResourceStrategy
from openllm._strategies import NvidiaGpuResource


def test_nvidia_gpu_resource_from_env(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 2
        assert resource == ["0", "1"]
        mcls.delenv("CUDA_VISIBLE_DEVICES")


def test_nvidia_gpu_cutoff_minus(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "0,2,-1,1")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 2
        assert resource == ["0", "2"]
        mcls.delenv("CUDA_VISIBLE_DEVICES")


def test_nvidia_gpu_neg_val(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "-1")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 0
        assert resource == []
        mcls.delenv("CUDA_VISIBLE_DEVICES")


def test_nvidia_gpu_parse_literal(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "GPU-5ebe9f43-ac33420d4628")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 1
        assert resource == ["GPU-5ebe9f43-ac33420d4628"]
        mcls.delenv("CUDA_VISIBLE_DEVICES")
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "GPU-5ebe9f43,GPU-ac33420d4628")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 2
        assert resource == ["GPU-5ebe9f43", "GPU-ac33420d4628"]
        mcls.delenv("CUDA_VISIBLE_DEVICES")
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "GPU-5ebe9f43,-1,GPU-ac33420d4628")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 1
        assert resource == ["GPU-5ebe9f43"]
        mcls.delenv("CUDA_VISIBLE_DEVICES")
    with monkeypatch.context() as mcls:
        mcls.setenv("CUDA_VISIBLE_DEVICES", "MIG-GPU-5ebe9f43-ac33420d4628")
        resource = NvidiaGpuResource.from_system()
        assert len(resource) == 1
        assert resource == ["MIG-GPU-5ebe9f43-ac33420d4628"]
        mcls.delenv("CUDA_VISIBLE_DEVICES")


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is not None, reason="skip GPUs test on CI")
def test_nvidia_gpu_validate(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mcls:
        # to make this tests works with system that has GPU
        mcls.setenv("CUDA_VISIBLE_DEVICES", "")
        assert len(NvidiaGpuResource.from_system()) >= 0  # TODO: real from_system tests

        assert pytest.raises(
            ValueError,
            NvidiaGpuResource.validate,
            [*NvidiaGpuResource.from_system(), 1],
        ).match("Input list should be all string type.")
        assert pytest.raises(ValueError, NvidiaGpuResource.validate, [-2]).match(
            "Input list should be all string type."
        )
        assert pytest.raises(ValueError, NvidiaGpuResource.validate, ["GPU-5ebe9f43", "GPU-ac33420d4628"]).match(
            "Failed to parse available GPUs UUID"
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="Test NVIDIA validation on Darwin only")
def test_nvidia_gpu_validation_on_darwin():
    assert pytest.raises(RuntimeError, NvidiaGpuResource.validate, ["0"]).match(
        "GPU is not available on Darwin system."
    )


def test_nvidia_gpu_from_spec(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mcls:
        # to make this tests works with system that has GPU
        mcls.setenv("CUDA_VISIBLE_DEVICES", "")
        assert NvidiaGpuResource.from_spec(1) == ["0"]
        assert NvidiaGpuResource.from_spec("5") == ["0", "1", "2", "3", "4"]
        assert NvidiaGpuResource.from_spec(1) == ["0"]
        assert NvidiaGpuResource.from_spec(2) == ["0", "1"]
        assert NvidiaGpuResource.from_spec("3") == ["0", "1", "2"]
        assert NvidiaGpuResource.from_spec([1, 3]) == ["1", "3"]
        assert NvidiaGpuResource.from_spec(["1", "3"]) == ["1", "3"]
        assert NvidiaGpuResource.from_spec(-1) == []
        assert NvidiaGpuResource.from_spec("-1") == []
        assert NvidiaGpuResource.from_spec("") == []
        assert NvidiaGpuResource.from_spec("-2") == []
        assert NvidiaGpuResource.from_spec("GPU-288347ab") == ["GPU-288347ab"]
        assert NvidiaGpuResource.from_spec("GPU-288347ab,-1,GPU-ac33420d4628") == ["GPU-288347ab"]
        assert NvidiaGpuResource.from_spec("GPU-288347ab,GPU-ac33420d4628") == ["GPU-288347ab", "GPU-ac33420d4628"]
        assert NvidiaGpuResource.from_spec("MIG-GPU-288347ab") == ["MIG-GPU-288347ab"]

    with pytest.raises(TypeError):
        NvidiaGpuResource.from_spec((1, 2, 3))
    with pytest.raises(TypeError):
        NvidiaGpuResource.from_spec(1.5)
    with pytest.raises(ValueError):
        assert NvidiaGpuResource.from_spec(-2)


class GPURunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "amd.com/gpu")


def unvalidated_get_resource(x: dict[str, t.Any], y: str, validate: bool = False):
    return get_resource(x, y, validate=validate)


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
    ).match("No known supported resource available for *")
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
