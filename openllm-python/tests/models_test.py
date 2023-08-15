from __future__ import annotations
import os
import typing as t

import pytest

if t.TYPE_CHECKING:
  import openllm

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is not None, reason="Model is too large for CI")
def test_flan_t5_implementation(prompt: str, llm: openllm.LLM[t.Any, t.Any]):
  assert llm(prompt)

  assert llm(prompt, temperature=0.8, top_p=0.23)

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is not None, reason="Model is too large for CI")
def test_opt_implementation(prompt: str, llm: openllm.LLM[t.Any, t.Any]):
  assert llm(prompt)

  assert llm(prompt, temperature=0.9, top_k=8)

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") is not None, reason="Model is too large for CI")
def test_baichuan_implementation(prompt: str, llm: openllm.LLM[t.Any, t.Any]):
  assert llm(prompt)

  assert llm(prompt, temperature=0.95)
