from __future__ import annotations
import shutil
import subprocess
import typing as t

import click
import psutil

from simple_di import Provide
from simple_di import inject

import bentoml

from bentoml._internal.configuration.containers import BentoMLContainer
from openllm.cli import termui
from openllm.cli._factory import bento_complete_envvar
from openllm.cli._factory import machine_option

if t.TYPE_CHECKING:
  from bentoml._internal.bento import BentoStore

@click.command('dive_bentos', context_settings=termui.CONTEXT_SETTINGS)
@click.argument('bento', type=str, shell_complete=bento_complete_envvar)
@machine_option
@click.pass_context
@inject
def cli(ctx: click.Context, bento: str, machine: bool, _bento_store: BentoStore = Provide[BentoMLContainer.bento_store]) -> str | None:
  '''Dive into a BentoLLM. This is synonymous to cd $(b get <bento>:<tag> -o path).'''
  try:
    bentomodel = _bento_store.get(bento)
  except bentoml.exceptions.NotFound:
    ctx.fail(f'Bento {bento} not found. Make sure to call `openllm build` first.')
  if 'bundler' not in bentomodel.info.labels or bentomodel.info.labels['bundler'] != 'openllm.bundle':
    ctx.fail(f"Bento is either too old or not built with OpenLLM. Make sure to use ``openllm build {bentomodel.info.labels['start_name']}`` for correctness.")
  if machine: return bentomodel.path
  # copy and paste this into a new shell
  if psutil.WINDOWS: subprocess.check_call([shutil.which('dir') or 'dir'], cwd=bentomodel.path)
  else: subprocess.check_call([shutil.which('ls') or 'ls', '-Rrthla'], cwd=bentomodel.path)
  ctx.exit(0)
