from __future__ import annotations

import asyncio
import contextlib
import types

import httpx
import pytest

from openllm import local


class _Proc:
  pid = 1
  stdout = None
  stderr = None

  def terminate(self) -> None:
    self.terminated = True


def test_run_model_sleeps_between_non_200_readyz_polls(monkeypatch):
  polls = 0
  sleeps: list[float] = []

  def fake_get(url, timeout=None):
    nonlocal polls
    polls += 1
    return httpx.Response(503, request=httpx.Request('GET', url))

  async def fake_sleep(delay):
    sleeps.append(delay)

  @contextlib.asynccontextmanager
  async def fake_async_run_command(*args, **kwargs):
    yield _Proc()

  monkeypatch.setattr(local, '_get_serve_cmd', lambda *a, **k: (['serve'], {}))
  monkeypatch.setattr(local, 'ensure_venv', lambda *a, **k: None)
  monkeypatch.setattr(local, 'async_run_command', fake_async_run_command)
  monkeypatch.setattr(local, 'output', lambda *a, **k: None)
  monkeypatch.setattr(local.httpx, 'get', fake_get)
  monkeypatch.setattr(local.asyncio, 'sleep', fake_sleep)

  bento = types.SimpleNamespace()
  asyncio.run(local._run_model(bento, port=3000, timeout=5))

  # Every failed poll must wait before the next one, not spin through the timeout.
  assert polls == 5
  assert sleeps == [1] * 5
