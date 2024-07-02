from __future__ import annotations

import attr
import time
import functools
import signal
import io
import click
import asyncio
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import sysconfig
import typing
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import typer
import typer.core

from bentoml._internal.utils.analytics import CliEvent
from bentoml._internal.utils.analytics import BENTOML_DO_NOT_TRACK as DO_NOT_TRACK

ERROR_STYLE = "red"
SUCCESS_STYLE = "green"


CLLAMA_HOME = pathlib.Path.home() / ".openllm_next"
REPO_DIR = CLLAMA_HOME / "repos"
TEMP_DIR = CLLAMA_HOME / "temp"
VENV_DIR = CLLAMA_HOME / "venv"

REPO_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True, parents=True)
VENV_DIR.mkdir(exist_ok=True, parents=True)

CONFIG_FILE = CLLAMA_HOME / "config.json"

CHECKED = "☆"

T = typing.TypeVar("T")


@attr.define
class OpenllmCliEvent(CliEvent):
    pass


class OrderedCommands(typer.core.TyperGroup):
    def list_commands(self, _: click.Context) -> typing.Iterable[str]:
        return list(self.commands)


class OpenLLMTyper(typer.Typer):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        no_args_is_help = kwargs.pop("no_args_is_help", True)
        context_settings = kwargs.pop("context_settings", {})
        if "help_option_names" not in context_settings:
            context_settings["help_option_names"] = ("-h", "--help")
        if "max_content_width" not in context_settings:
            context_settings["max_content_width"] = int(
                os.environ.get("COLUMNS", str(120))
            )
        klass = kwargs.pop("cls", OrderedCommands)

        super().__init__(
            *args,
            cls=klass,
            no_args_is_help=no_args_is_help,
            context_settings=context_settings,
            **kwargs,
        )

    def command(self, *args: typing.Any, **kwargs: typing.Any):
        def decorator(f):
            @functools.wraps(f)
            @click.pass_context
            def wrapped(ctx: click.Context, *args, **kwargs):
                from bentoml._internal.utils.analytics import track

                do_not_track = (
                    os.environ.get(DO_NOT_TRACK, str(False)).lower() == "true"
                )

                # so we know that the root program is openllm
                command_name = ctx.info_name
                if ctx.parent.parent is not None:
                    # openllm model list
                    command_group = ctx.parent.info_name
                elif ctx.parent.info_name == ctx.find_root().info_name:
                    # openllm run
                    command_group = "openllm"

                if do_not_track:
                    return f(*args, **kwargs)
                start_time = time.time_ns()
                try:
                    return_value = f(*args, **kwargs)
                    duration_in_ns = time.time_ns() - start_time
                    track(
                        OpenllmCliEvent(
                            cmd_group=command_group,
                            cmd_name=command_name,
                            duration_in_ms=duration_in_ns / 1e6,
                        )
                    )
                    return return_value
                except BaseException as e:
                    duration_in_ns = time.time_ns() - start_time
                    track(
                        OpenllmCliEvent(
                            cmd_group=command_group,
                            cmd_name=command_name,
                            duration_in_ms=duration_in_ns / 1e6,
                            error_type=type(e).__name__,
                            return_code=2 if isinstance(e, KeyboardInterrupt) else 1,
                        )
                    )
                    raise

            return typer.Typer.command(self, *args, **kwargs)(wrapped)

        return decorator


class ContextVar(typing.Generic[T]):
    def __init__(self, default: T):
        self._stack: list[T] = []
        self._default = default

    def get(self) -> T:
        if self._stack:
            return self._stack[-1]
        return self._default

    def set(self, value):
        self._stack.append(value)

    @contextmanager
    def patch(self, value):
        self._stack.append(value)
        try:
            yield
        finally:
            self._stack.pop()


VERBOSE_LEVEL = ContextVar(10)
INTERACTIVE = ContextVar(False)
FORCE = ContextVar(False)


def output(content, level=0, style=None, end=None):
    import questionary

    if level > VERBOSE_LEVEL.get():
        return

    if not isinstance(content, str):
        import pyaml

        out = io.StringIO()
        pyaml.pprint(
            content,
            dst=out,
            sort_dicts=False,
            sort_keys=False,
        )
        questionary.print(out.getvalue(), style=style, end="" if end is None else end)
        out.close()

    if isinstance(content, str):
        questionary.print(content, style=style, end="\n" if end is None else end)


class Config(SimpleNamespace):
    repos: dict[str, str] = {
        "default": "git+https://github.com/bentoml/openllm-repo@main"
    }
    default_repo: str = "default"


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return Config(**json.load(f))
    return Config()


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.dict(), f, indent=2)


class RepoInfo(SimpleNamespace):
    name: str
    path: pathlib.Path
    url: str
    server: str
    owner: str
    repo: str
    branch: str

    def tolist(self):
        if VERBOSE_LEVEL.get() <= 0:
            return f"{self.name} ({self.url})"
        if VERBOSE_LEVEL.get() <= 10:
            return dict(
                name=self.name,
                url=self.url,
                path=str(self.path),
            )
        if VERBOSE_LEVEL.get() <= 20:
            return dict(
                name=self.name,
                url=self.url,
                path=str(self.path),
                server=self.server,
                owner=self.owner,
                repo=self.repo,
                branch=self.branch,
            )


class BentoInfo(SimpleNamespace):
    repo: RepoInfo
    path: pathlib.Path

    def __str__(self):
        if self.repo.name == "default":
            return f"{self.tag}"
        else:
            return f"{self.repo.name}/{self.tag}"

    def __hash__(self):
        return md5(str(self.path))

    @property
    def tag(self) -> str:
        return f"{self.path.parent.name}:{self.path.name}"

    @property
    def name(self) -> str:
        return self.path.parent.name

    @property
    def version(self) -> str:
        return self.path.name

    @functools.cached_property
    def bento_yaml(self) -> dict:
        import yaml

        bento_file = self.path / "bento.yaml"
        return yaml.safe_load(bento_file.read_text())

    @functools.cached_property
    def platforms(self) -> list[str]:
        return self.bento_yaml["labels"].get("platforms", "linux").split(",")

    @functools.cached_property
    def pretty_yaml(self) -> dict:
        def _pretty_routes(routes):
            return {
                route["route"]: {
                    "input": {
                        k: v["type"] for k, v in route["input"]["properties"].items()
                    },
                    "output": route["output"]["type"],
                }
                for route in routes
            }

        if len(self.bento_yaml["services"]) == 1:
            pretty_yaml = {
                "apis": _pretty_routes(self.bento_yaml["schema"]["routes"]),
                "resources": self.bento_yaml["services"][0]["config"]["resources"],
                "envs": self.bento_yaml["envs"],
                "platforms": self.platforms,
            }
            return pretty_yaml
        return self.bento_yaml

    @functools.cached_property
    def pretty_accelerator(self) -> str:
        from openllm_next.accelerator_spec import ACCELERATOR_SPECS

        try:
            resources = self.bento_yaml["services"][0]["config"]["resources"]
            if resources["gpu"] > 0:
                acc = ACCELERATOR_SPECS[resources["gpu_type"]]
                return f"{acc.memory_size:.0f}GB x{resources['gpu']} ({acc.model})"
            return ""
        except KeyError:
            return ""

    def tolist(self):
        verbose = VERBOSE_LEVEL.get()
        if verbose <= 0:
            return str(self)
        if verbose <= 10:
            return dict(
                tag=self.tag,
                repo=self.repo.tolist(),
                path=str(self.path),
                model_card=self.pretty_yaml,
            )
        if verbose <= 20:
            return dict(
                tag=self.tag,
                repo=self.repo.tolist(),
                path=str(self.path),
                bento_yaml=self.bento_yaml,
            )


class VenvSpec(SimpleNamespace):
    python_version: str
    python_packages: dict[str, str]
    name_prefix = ""

    def __hash__(self):
        return md5(
            # self.python_version,
            *sorted(self.python_packages.values()),
        )


class Accelerator(SimpleNamespace):
    model: str
    memory_size: float

    def __gt__(self, other):
        return self.memory_size > other.memory_size

    def __eq__(self, other):
        return self.memory_size == other.memory_size


class DeploymentTarget(SimpleNamespace):
    source: str = "local"
    name: str = "local"
    price: str = ""
    platform = "linux"
    accelerators: list[Accelerator]

    def __hash__(self):
        return hash(self.source)

    @property
    def accelerators_repr(self) -> str:
        accs = {a.model for a in self.accelerators}
        if len(accs) == 0:
            return "null"
        if len(accs) == 1:
            a = self.accelerators[0]
            return f"{a.model} x{len(self.accelerators)}"
        return ", ".join((f"{a.model}" for a in self.accelerators))


def run_command(
    cmd,
    cwd=None,
    env=None,
    copy_env=True,
    venv=None,
    silent=False,
) -> subprocess.CompletedProcess:
    import shlex

    env = env or {}
    cmd = [str(c) for c in cmd]
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    if not silent:
        output("\n")
        if cwd:
            output(f"$ cd {cwd}", style="bold")
        if env:
            for k, v in env.items():
                output(f"$ export {k}={shlex.quote(v)}", style="bold")
        if venv:
            output(f"$ source {venv / 'bin' / 'activate'}", style="bold")
        output(f"$ {' '.join(cmd)}", style="bold")

    if venv:
        py = venv / bin_dir / f"python{sysconfig.get_config_var('EXE')}"
    else:
        py = sys.executable

    if copy_env:
        env = {**os.environ, **env}

    if cmd and cmd[0] == "bentoml":
        cmd = [py, "-m", "bentoml"] + cmd[1:]
    if cmd and cmd[0] == "python":
        cmd = [py] + cmd[1:]

    try:
        if silent:
            return subprocess.run(  # type: ignore
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            return subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
            )
    except subprocess.CalledProcessError:
        output("Command failed", style="red")
        raise typer.Exit(1)


async def stream_command_output(stream, style="gray"):
    async for line in stream:
        output(line.decode(), style=style, end="")


@asynccontextmanager
async def async_run_command(
    cmd,
    cwd=None,
    env=None,
    copy_env=True,
    venv=None,
    silent=True,
):
    import shlex

    env = env or {}
    cmd = [str(c) for c in cmd]

    if not silent:
        output("\n")
        if cwd:
            output(f"$ cd {cwd}", style="bold")
        if env:
            for k, v in env.items():
                output(f"$ export {k}={shlex.quote(v)}", style="bold")
        if venv:
            output(f"$ source {venv / 'bin' / 'activate'}", style="bold")
        output(f"$ {' '.join(cmd)}", style="bold")

    if venv:
        py = venv / "bin" / "python"
    else:
        py = sys.executable

    if copy_env:
        env = {**os.environ, **env}

    if cmd and cmd[0] == "bentoml":
        cmd = [py, "-m", "bentoml"] + cmd[1:]
    if cmd and cmd[0] == "python":
        cmd = [py] + cmd[1:]

    proc = None
    try:
        proc = await asyncio.create_subprocess_shell(
            " ".join(map(str, cmd)),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        yield proc
    except subprocess.CalledProcessError:
        output("Command failed", style="red")
        raise typer.Exit(1)
    finally:
        if proc:
            proc.send_signal(signal.SIGINT)
            await proc.wait()


def md5(*strings: str) -> int:
    m = hashlib.md5()
    for s in strings:
        m.update(s.encode())
    return int(m.hexdigest(), 16)
