from __future__ import annotations

import typing as t

from ..utils import DummyMetaclass, require_backends


class FlaxFlanT5(metaclass=DummyMetaclass):
    _backends = ["flax"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["flax"])


class AutoFlaxLLM(metaclass=DummyMetaclass):
    _backends = ["flax"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["flax"])


MODEL_FLAX_MAPPING_NAMES = None

MODEL_FLAX_MAPPING = None
