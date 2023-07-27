from __future__ import annotations
import shutil
import subprocess
import sys
import typing as t

import click
import psutil
from simple_di import Provide
from simple_di import inject

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer

from .. import termui

# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if sys.version_info[:2] >= (3, 11):
    from typing import overload
else:
    from typing_extensions import overload

if t.TYPE_CHECKING:
    from bentoml._internal.bento import BentoStore

@overload
def cli(ctx: click.Context, bento: str, machine: t.Literal[True] = True, _bento_store: BentoStore = ...) -> str: ...
@overload
def cli(ctx: click.Context, bento: str, machine: t.Literal[False] = ..., _bento_store: BentoStore = ...) -> None: ...
@click.command("dive_bentos", context_settings=termui.CONTEXT_SETTINGS)
@click.argument("bento", type=str)
@click.option("--machine", is_flag=True, default=False, hidden=True)
@click.pass_context
@inject
def cli(ctx: click.Context, bento: str, machine: bool, _bento_store: BentoStore = Provide[BentoMLContainer.bento_store]) -> str | None:
    """Dive into a BentoLLM. This is synonymous to cd $(b get <bento>:<tag> -o path)."""
    try: bentomodel = _bento_store.get(bento)
    except bentoml.exceptions.NotFound: ctx.fail(f"Bento {bento} not found. Make sure to call `openllm build first`")
    if "bundler" not in  bentomodel.info.labels or bentomodel.info.labels["bundler"] != "openllm.bundle": ctx.fail(f"Bento is either too old or not built with OpenLLM. Make sure to use ``openllm build {bentomodel.info.labels['start_name']}`` for correctness.")
    if machine: return bentomodel.path
    # copy and paste this into a new shell
    if psutil.WINDOWS: subprocess.check_output([shutil.which("dir") or "dir"], cwd=bentomodel.path)
    else:subprocess.check_output([shutil.which("ls") or "ls", "-R"], cwd=bentomodel.path)
    ctx.exit(0)
