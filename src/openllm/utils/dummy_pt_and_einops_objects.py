from __future__ import annotations

import typing as t

from ..utils import DummyMetaclass, require_backends


class Falcon(metaclass=DummyMetaclass):
    _backends = ["torch", "einops"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["torch", "einops"])
