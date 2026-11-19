from __future__ import annotations

import pathlib

import pytest

from openllm import venv
from openllm.common import EnvVars, VenvSpec


def test_venv_spec_hash_includes_python_version() -> None:
  spec = VenvSpec(python_version='3.11', requirements_txt='example==1.0', envs=EnvVars())
  other = spec.model_copy(update={'python_version': '3.12'})

  assert hash(spec) != hash(other)


def test_venv_spec_hash_preserves_requirement_normalization() -> None:
  spec = VenvSpec(python_version='3.11', requirements_txt='b==2.0\na==1.0', envs=EnvVars())
  other = spec.model_copy(update={'requirements_txt': '# comment\na==1.0\n\nb==2.0\n'})

  assert hash(spec) == hash(other)


def test_ensure_venv_separates_python_versions(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(venv, 'VENV_DIR', tmp_path)
  monkeypatch.setattr(venv, 'output', lambda *args, **kwargs: None)
  commands: list[list[str]] = []

  def run_command(cmd: list[str], **kwargs: object) -> None:
    commands.append(cmd)
    if cmd[2:4] == ['uv', 'venv']:
      pathlib.Path(cmd[4]).mkdir()

  monkeypatch.setattr(venv, 'run_command', run_command)
  first_spec = VenvSpec(python_version='3.11', requirements_txt='example==1.0', envs=EnvVars())
  second_spec = first_spec.model_copy(update={'python_version': '3.12'})

  first_path = venv._ensure_venv(first_spec)
  commands.clear()
  second_path = venv._ensure_venv(second_spec)

  assert first_path != second_path
  assert commands[0][-2:] == ['-p', '3.12']
  assert (first_path / 'DONE').exists()
  assert (second_path / 'DONE').exists()

  commands.clear()
  assert venv._ensure_venv(second_spec) == second_path
  assert commands == []
