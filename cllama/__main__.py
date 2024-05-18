import typer
import shlex
import os
from typing_extensions import TypedDict
import collections

import prompt_toolkit
import shutil
import pydantic
import yaml
import json
import questionary
import re
import subprocess
import pyaml
import pathlib
from cllama.spec import GPU_MEMORY


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


app = typer.Typer()
repo_app = typer.Typer()
model_app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")


class Config(pydantic.BaseModel):
    repos: dict[str, str] = {
        "default": "git+https://github.com/bojiang/bentovllm@main#subdirectory=bentoml"
    }
    default_repo: str = "default"


def _load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return Config(**json.load(f))
    return Config()


def _save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.dict(), f, indent=2)


class RepoInfo(TypedDict):
    name: str
    path: str
    url: str
    server: str
    owner: str
    repo: str
    branch: str


class ModelInfo(TypedDict):
    repo: RepoInfo
    path: str


class BentoInfo(TypedDict):
    model: ModelInfo
    bento_yaml: dict


def _load_model_map() -> dict[str, dict[str, ModelInfo]]:
    model_map = collections.defaultdict(dict)
    config = _load_config()
    for repo_name, repo_url in config.repos.items():
        server, owner, repo, branch = _parse_repo_url(repo_url)
        repo_dir = REPO_DIR / server / owner / repo
        for path in repo_dir.glob("bentoml/bentos/*/*"):
            if path.is_dir():
                model_map[path.parent.name][path.name] = ModelInfo(
                    repo=RepoInfo(
                        name=repo_name,
                        url=repo_url,
                        server=server,
                        owner=owner,
                        repo=repo,
                        branch=branch,
                        path=str(repo_dir),
                    ),
                    path=str(path),
                )
            elif path.is_file():
                with open(path) as f:
                    origin_name = f.read().strip()
                origin_path = path.parent / origin_name
                model_map[path.parent.name][path.name] = ModelInfo(
                    repo=RepoInfo(
                        name=repo_name,
                        url=repo_url,
                        server=server,
                        owner=owner,
                        repo=repo,
                        branch=branch,
                        path=str(repo_dir),
                    ),
                    path=str(origin_path),
                )
    return model_map


GIT_REPO_RE = re.compile(
    r"git\+https://(?P<server>.+)/(?P<owner>.+)/(?P<repo>.+?)(@(?P<branch>.+))?$"
)


@repo_app.command(name="list")
def repo_list():
    config = _load_config()
    pyaml.pprint(config.repos)


def _parse_repo_url(repo_url):
    """
    parse the git repo url to server, owner, repo name, branch
    >>> _parse_repo_url("git+https://github.com/bojiang/bentovllm@main")
    ('github.com', 'bojiang', 'bentovllm', 'main')

    >>> _parse_repo_url("git+https://github.com/bojiang/bentovllm")
    ('github.com', 'bojiang', 'bentovllm', 'main')
    """
    match = GIT_REPO_RE.match(repo_url)
    if not match:
        raise ValueError(f"Invalid git repo url: {repo_url}")
    return (
        match.group("server"),
        match.group("owner"),
        match.group("repo"),
        match.group("branch") or "main",
    )


@repo_app.command(name="add")
def repo_add(name: str, repo: str):
    name = name.lower()
    if not name.isidentifier():
        questionary.print(
            f"Invalid repo name: {name}, should only contain letters, numbers and underscores",
            style=ERROR_STYLE,
        )
        return

    config = _load_config()
    if name in config.repos:
        override = questionary.confirm(
            f"Repo {name} already exists({config.repos[name]}), override?"
        ).ask()
        if not override:
            return

    config.repos[name] = repo
    _save_config(config)
    pyaml.pprint(config.repos)


@repo_app.command(name="remove")
def repo_remove(name: str):
    config = _load_config()
    if name not in config.repos:
        questionary.print(f"Repo {name} does not exist", style=ERROR_STYLE)
        return

    del config.repos[name]
    _save_config(config)
    pyaml.pprint(config.repos)


def _run_command(cmd, cwd=None, env=None, copy_env=True):
    questionary.print("\n")
    env = env or {}
    if cwd:
        questionary.print(f"$ cd {cwd}", style="bold")
    if env:
        for k, v in env.items():
            questionary.print(f"$ export {k}={shlex.quote(v)}", style="bold")
    if copy_env:
        env = {**os.environ, **env}
    questionary.print(f"$ {' '.join(cmd)}", style="bold")
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError:
        questionary.print("Command failed", style=ERROR_STYLE)
        return


@repo_app.command(name="update")
def repo_update():
    config = _load_config()
    repos_in_use = set()
    for name, repo in config.repos.items():
        server, owner, repo_name, branch = _parse_repo_url(repo)
        repos_in_use.add((server, owner, repo_name))
        repo_dir = REPO_DIR / server / owner / repo_name
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                cmd = [
                    "git",
                    "clone",
                    "--branch",
                    branch,
                    f"https://{server}/{owner}/{repo_name}.git",
                    str(repo_dir),
                ]
                _run_command(cmd)
            except subprocess.CalledProcessError:
                shutil.rmtree(repo_dir, ignore_errors=True)
                questionary.print(f"Failed to clone repo {name}", style=ERROR_STYLE)
        else:
            try:
                cmd = ["git", "fetch", "origin", branch]
                _run_command(cmd, cwd=repo_dir)
                cmd = ["git", "reset", "--hard", f"origin/{branch}"]
                _run_command(cmd, cwd=repo_dir)
            except:
                shutil.rmtree(repo_dir, ignore_errors=True)
                questionary.print(f"Failed to update repo {name}", style=ERROR_STYLE)
    for repo_dir in REPO_DIR.glob("*/*/*"):
        if tuple(repo_dir.parts[-3:]) not in repos_in_use:
            shutil.rmtree(repo_dir, ignore_errors=True)
            questionary.print(f"Removed unused repo {repo_dir}")
    questionary.print("Repos updated", style=SUCCESS_STYLE)


@model_app.command(name="list")
def model_list():
    pyaml.pprint(_load_model_map())


def _get_bento_info(tag):
    model_map = _load_model_map()
    bento, version = tag.split(":")
    if bento not in model_map or version not in model_map[bento]:
        questionary.print(f"Model {tag} not found", style=ERROR_STYLE)
        return
    model_info = model_map[bento][version]
    path = pathlib.Path(model_info["path"])

    bento_file = path / "bento.yaml"
    bento_info = yaml.safe_load(bento_file.read_text())
    return BentoInfo(
        model=model_info,
        bento_yaml=bento_info,
    )


@model_app.command(name="get")
def model_get(tag: str):
    bento_info = _get_bento_info(tag)
    if bento_info:
        pyaml.pprint(bento_info)


def _serve_model(model: str):
    if ":" not in model:
        model = f"{model}:latest"
    bento_info = _get_bento_info(model)
    if not bento_info:
        questionary.print(f"Model {model} not found", style=ERROR_STYLE)
        return
    cmd = ["bentoml", "serve", model]
    env = {
        "CLLAMA_MODEL": model,
        "BENTOML_HOME": bento_info["model"]["repo"]["path"] + "/bentoml",
    }
    _run_command(cmd, env=env)


@app.command()
def serve(model: str):
    _serve_model(model)


if __name__ == "__main__":
    app()
