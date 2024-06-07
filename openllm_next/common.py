import functools
import signal
import asyncio
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import typing
from contextlib import contextmanager, asynccontextmanager
from types import SimpleNamespace

import questionary
import typer

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


T = typing.TypeVar("T")


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


VERBOSE_LEVEL = ContextVar(0)


class Config(SimpleNamespace):
    repos: dict[str, str] = {
        "default": "git+https://github.com/bojiang/openllm-repo@main"
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
        if VERBOSE_LEVEL.get() <= 1:
            return dict(
                name=self.name,
                url=self.url,
                path=str(self.path),
            )
        if VERBOSE_LEVEL.get() <= 2:
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
            }
            return pretty_yaml
        return self.bento_yaml

    def tolist(self):
        verbose = VERBOSE_LEVEL.get()
        if verbose <= 0:
            return str(self)
        if verbose <= 1:
            return dict(
                tag=self.tag,
                repo=self.repo.tolist(),
                path=str(self.path),
                model_card=self.pretty_yaml,
            )
        if verbose <= 2:
            return dict(
                tag=self.tag,
                repo=self.repo.tolist(),
                path=str(self.path),
                bento_yaml=self.bento_yaml,
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
    if not silent:
        questionary.print("\n")
        if cwd:
            questionary.print(f"$ cd {cwd}", style="bold")
        if env:
            for k, v in env.items():
                questionary.print(f"$ export {k}={shlex.quote(v)}", style="bold")
        if venv:
            questionary.print(f"$ source {venv / 'bin' / 'activate'}", style="bold")
        questionary.print(f"$ {' '.join(cmd)}", style="bold")

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
        questionary.print("Command failed", style=ERROR_STYLE)
        raise typer.Exit(1)


async def stream_command_output(stream, style="gray"):
    async for line in stream:
        questionary.print(line.decode(), style=style, end="")


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
        questionary.print("\n")
        if cwd:
            questionary.print(f"$ cd {cwd}", style="bold")
        if env:
            for k, v in env.items():
                questionary.print(f"$ export {k}={shlex.quote(v)}", style="bold")
        if venv:
            questionary.print(f"$ source {venv / 'bin' / 'activate'}", style="bold")
        questionary.print(f"$ {' '.join(cmd)}", style="bold")

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
        questionary.print("Command failed", style=ERROR_STYLE)
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
