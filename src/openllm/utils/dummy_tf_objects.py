from __future__ import annotations

import typing as t

from ..utils import DummyMetaclass, require_backends


class TFFlanT5(metaclass=DummyMetaclass):
    _backends = ["tf"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["tf"])


class AutoTFLLM(metaclass=DummyMetaclass):
    _backends = ["tf"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["tf"])


MODEL_TF_MAPPING_NAMES = None

MODEL_TF_MAPPING = None
