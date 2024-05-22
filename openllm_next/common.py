import sys
import os
import typing
from types import SimpleNamespace
import json
from contextlib import contextmanager
import questionary
import subprocess
import pathlib


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
    cache_path: pathlib.Path
    url: str
    server: str
    owner: str
    repo: str
    branch: str

    def tolist(self):
        if VERBOSE_LEVEL.get() <= 0:
            return self.name
        if VERBOSE_LEVEL.get() <= 2:
            return dict(
                name=self.name,
                cache_path=str(self.cache_path),
                url=self.url,
                server=self.server,
                owner=self.owner,
                repo=self.repo,
                branch=self.branch,
            )


class ModelInfo(SimpleNamespace):
    repo: RepoInfo
    path: str

    def tolist(self):
        if VERBOSE_LEVEL.get() <= 0:
            return f"{self.repo.name}"
        if VERBOSE_LEVEL.get() <= 2:
            return dict(
                repo=self.repo.tolist(),
                path=self.path,
            )


class BentoInfo(SimpleNamespace):
    model: ModelInfo
    bento_yaml: dict

    def tolist(self):
        if VERBOSE_LEVEL.get() <= 0:
            return f"{self.model.repo.name}"
        if VERBOSE_LEVEL.get() <= 2:
            return dict(
                model=self.model.tolist(),
                bento_yaml=self.bento_yaml,
            )


def run_command(
    cmd,
    cwd=None,
    env=None,
    copy_env=True,
    silent=False,
    check=True,
) -> subprocess.CompletedProcess | subprocess.Popen | None:
    import shlex

    env = env or {}
    if not silent:
        questionary.print("\n")
        if cwd:
            questionary.print(f"$ cd {cwd}", style="bold")
        if env:
            for k, v in env.items():
                questionary.print(f"$ export {k}={shlex.quote(v)}", style="bold")
        questionary.print(f"> {' '.join(cmd)}", style="bold")
    if copy_env:
        env = {**os.environ, **env}
    if cmd and cmd[0] == "bentoml":
        cmd = [sys.executable, "-m", "bentoml"] + cmd[1:]
    if cmd and cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
    try:
        if silent:
            return subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                check=check,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            return subprocess.run(cmd, cwd=cwd, env=env, check=check)
    except subprocess.CalledProcessError:
        questionary.print("Command failed", style=ERROR_STYLE)
        return None
