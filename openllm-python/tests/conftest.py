from __future__ import annotations
import itertools
import os
import typing as t

import pytest

import openllm

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import LiteralBackend

_MODELING_MAPPING = {'flan_t5': 'google/flan-t5-small', 'opt': 'facebook/opt-125m', 'baichuan': 'baichuan-inc/Baichuan-7B'}
_TRUST_REMOTE_MAPPING = {'flan_t5': False, 'opt': False, 'baichuan': True}
_PROMPT_MAPPING = {'qa': 'Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?'}


def parametrise_local_llm(model: str) -> t.Generator[tuple[str, openllm.LLM[t.Any, t.Any]], None, None]:
  if model not in _MODELING_MAPPING:
    pytest.skip(f"'{model}' is not yet supported in framework testing.")
  backends: tuple[LiteralBackend, ...] = ('pt',)
  for backend, prompt in itertools.product(backends, _PROMPT_MAPPING.keys()):
    yield prompt, openllm.LLM(_MODELING_MAPPING[model], backend=backend, trust_remote_code=_TRUST_REMOTE_MAPPING[model])


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
  if os.getenv('GITHUB_ACTIONS') is None:
    if 'prompt' in metafunc.fixturenames and 'llm' in metafunc.fixturenames:
      metafunc.parametrize('prompt,llm', [(p, llm) for p, llm in parametrise_local_llm(metafunc.function.__name__[5:-15])])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
  # If no tests are collected, pytest exists with code 5, which makes the CI fail.
  if exitstatus == 5:
    session.exitstatus = 0
