from __future__ import annotations
import typing as t

from ..utils import DummyMetaclass
from ..utils import require_backends


class AutoVLLM(metaclass=DummyMetaclass):
    _backends = ["vllm"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["vllm"])


MODEL_VLLM_MAPPING = None
