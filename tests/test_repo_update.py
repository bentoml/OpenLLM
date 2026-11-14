from __future__ import annotations

import importlib
import pathlib
import sys
import types
import typing

import pytest
from typer.testing import CliRunner


@pytest.fixture
def repo_mod(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[types.ModuleType, types.ModuleType]:
  monkeypatch.setenv('OPENLLM_HOME', str(tmp_path))
  monkeypatch.delenv('OPENLLM_TEST_REPO', raising=False)
  # the CLI wrapper only needs bentoml's analytics.track, stub it so bentoml is not required
  monkeypatch.setenv('BENTOML_DO_NOT_TRACK', 'True')
  for name in ['bentoml', 'bentoml._internal', 'bentoml._internal.utils']:
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
  analytics = types.ModuleType('bentoml._internal.utils.analytics')
  setattr(analytics, 'track', lambda event: None)
  monkeypatch.setitem(sys.modules, 'bentoml._internal.utils.analytics', analytics)
  for name in ['openllm.common', 'openllm.repo']:
    sys.modules.pop(name, None)
  common = importlib.import_module('openllm.common')
  repo = importlib.import_module('openllm.repo')
  monkeypatch.setattr(repo, 'output', lambda *a, **k: None)
  monkeypatch.setattr(repo, '_complete_alias', lambda name: None)
  return common, repo


@pytest.mark.parametrize('branch', ['main', 'release/1.0', 'a/b/c'])
def test_update_keeps_clone_of_configured_branch(
  repo_mod: tuple[types.ModuleType, types.ModuleType], monkeypatch: pytest.MonkeyPatch, branch: str
) -> None:
  common, repo = repo_mod
  config = common.load_config()
  config.repos = {'mine': f'https://github.com/acme/models@{branch}'}
  common.save_config(config)

  def fake_clone(info: typing.Any) -> None:
    info.path.mkdir(parents=True)
    (info.path / 'marker').write_text('x')

  monkeypatch.setattr(repo, '_clone_repo', fake_clone)

  result = CliRunner().invoke(repo.app, ['update'])
  assert result.exit_code == 0, result.output
  info = repo.parse_repo_url(f'https://github.com/acme/models@{branch}')
  assert (info.path / 'marker').exists()


def test_update_removes_unused_cache(
  repo_mod: tuple[types.ModuleType, types.ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
  common, repo = repo_mod
  config = common.load_config()
  config.repos = {'mine': 'https://github.com/acme/models@release/1.0'}
  common.save_config(config)
  stale = common.REPO_DIR / 'github.com' / 'old' / 'models' / 'main'
  stale.mkdir(parents=True)
  monkeypatch.setattr(repo, '_clone_repo', lambda info: info.path.mkdir(parents=True))

  assert CliRunner().invoke(repo.app, ['update']).exit_code == 0
  assert not stale.exists()
