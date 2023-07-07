from __future__ import annotations

import openllm
import typing as t


def test_runtime_impl(prompt: str, llm: openllm.LLM[t.Any, t.Any]):
    assert llm(prompt)

    assert llm(prompt, temperature=0.8, top_p=0.23)
