from __future__ import annotations

import typing as t

from ..utils import DummyMetaclass, require_backends


class ChatGLM(metaclass=DummyMetaclass):
    _backends = ["torch", "cpm_kernels"]

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        require_backends(self, ["torch", "cpm_kernels"])
