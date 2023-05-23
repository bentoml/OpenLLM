from __future__ import annotations

import typing as t

from ..utils import DummyMetaclass, require_backends


class FlanT5(metaclass=DummyMetaclass):
    _backends = ["torch"]

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        require_backends(self, ["torch"])


class DollyV2(metaclass=DummyMetaclass):
    _backends = ["torch"]

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        require_backends(self, ["torch"])


class AutoLLM(metaclass=DummyMetaclass):
    _backends = ["torch"]

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        require_backends(self, ["torch"])


MODEL_MAPPING_NAMES = None

MODEL_MAPPING = None
