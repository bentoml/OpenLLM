import pathlib

import pytest
from typer.testing import CliRunner

from openllm import clean


def test_clean_configs_removes_config_file(
  tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  config_file = tmp_path / 'config.json'
  config_file.write_text('{"default_repo": "custom"}')
  monkeypatch.setattr(clean, 'CONFIG_FILE', config_file)
  monkeypatch.setenv('BENTOML_DO_NOT_TRACK', 'true')

  result = CliRunner().invoke(clean.app, ['configs'])

  assert result.exit_code == 0
  assert not config_file.exists()
