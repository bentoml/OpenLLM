import functools
import os
import pathlib
import shutil
import typing
from typing import Iterable

import typer

from openllm.common import VENV_DIR, VERBOSE_LEVEL, BentoInfo, VenvSpec, output, run_command


@functools.lru_cache
def _resolve_packages(requirement: typing.Union[pathlib.Path, str]):
  from pip_requirements_parser import RequirementsFile

  requirements_txt = RequirementsFile.from_file(str(requirement), include_nested=True)
  return requirements_txt.requirements


def _filter_preheat_packages(requirements: Iterable) -> list[str]:
  PREHEAT_PIP_PACKAGES = ['torch', 'vllm']

  deps: list[str] = []
  for req in requirements:
    if req.is_editable or req.is_local_path or req.is_url or req.is_wheel or not req.name or not req.specifier:
      continue
    for sp in req.specifier:
      if sp.operator == '==' and req.name in PREHEAT_PIP_PACKAGES:
        assert req.line is not None
        deps.append(req.line)
        break
  return deps


@functools.lru_cache
def _resolve_bento_env_specs(bento: BentoInfo):
  ver_file = bento.path / 'env' / 'python' / 'version.txt'
  assert ver_file.exists(), f'cannot find version file in {bento.path}'

  lock_file = bento.path / 'env' / 'python' / 'requirements.lock.txt'
  if not lock_file.exists():
    lock_file = bento.path / 'env' / 'python' / 'requirements.txt'

  reqs = _resolve_packages(lock_file)
  preheat_packages = _filter_preheat_packages(reqs)
  ver = ver_file.read_text().strip()
  return (
    VenvSpec(python_version=ver, python_packages=preheat_packages, name_prefix=f"{bento.tag.replace(':', '_')}-1-"),
    VenvSpec(
      python_version=ver, python_packages=[v.line for v in reqs], name_prefix=f"{bento.tag.replace(':', '_')}-2-"
    ),
  )


def _get_lib_dir(venv: pathlib.Path) -> pathlib.Path:
  if os.name == 'nt':
    return venv / 'Lib/site-packages'
  else:
    return next(venv.glob('lib/python*')) / 'site-packages'


def _ensure_venv(env_spec: VenvSpec, parrent_venv: typing.Optional[pathlib.Path] = None) -> pathlib.Path:
  venv = VENV_DIR / str(hash(env_spec))
  if venv.exists() and not (venv / 'DONE').exists():
    shutil.rmtree(venv, ignore_errors=True)
  if not venv.exists():
    output(f'Installing model dependencies({venv})...', style='green')

    venv_py = venv / 'Scripts' / 'python.exe' if os.name == 'nt' else venv / 'bin' / 'python'
    try:
      run_command(['python', '-m', 'uv', 'venv', venv], silent=VERBOSE_LEVEL.get() < 10)
      lib_dir = _get_lib_dir(venv)
      if parrent_venv is not None:
        parent_lib_dir = _get_lib_dir(parrent_venv)
        with open(lib_dir / f'{parrent_venv.name}.pth', 'w+') as f:
          f.write(str(parent_lib_dir))
      with open(venv / 'requirements.txt', 'w') as f:
        f.write('\n'.join(sorted(env_spec.python_packages)))
      run_command(
        ['python', '-m', 'uv', 'pip', 'install', '-p', str(venv_py), '-r', venv / 'requirements.txt'],
        silent=VERBOSE_LEVEL.get() < 10,
      )
      with open(venv / 'DONE', 'w') as f:
        f.write('DONE')
    except Exception:
      shutil.rmtree(venv, ignore_errors=True)
      output(f'Failed to install dependencies to {venv}. Cleaned up.', style='red')
      raise typer.Exit(1)
    output(f'Successfully installed dependencies to {venv}.', style='green')
    return venv
  else:
    return venv


def _ensure_venvs(env_spec_list: Iterable[VenvSpec]) -> pathlib.Path:
  last_venv = None
  for env_spec in env_spec_list:
    last_venv = _ensure_venv(env_spec, last_venv)
  assert last_venv is not None
  return last_venv


def ensure_venv(bento: BentoInfo) -> pathlib.Path:
  return _ensure_venvs(_resolve_bento_env_specs(bento))


def _check_venv(env_spec: VenvSpec) -> bool:
  venv = VENV_DIR / str(hash(env_spec))
  if not venv.exists():
    return False
  if venv.exists() and not (venv / 'DONE').exists():
    return False
  return True


def check_venv(bento: BentoInfo) -> bool:
  return all(_check_venv(env_spec) for env_spec in _resolve_bento_env_specs(bento))
