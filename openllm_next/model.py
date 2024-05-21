import typer
import os
import collections

import yaml
import questionary
import pyaml
import pathlib
from openllm_next.common import (
    ERROR_STYLE,
    REPO_DIR,
    load_config,
    ModelInfo,
    RepoInfo,
    BentoInfo,
)
from openllm_next.repo import parse_repo_url


app = typer.Typer()


def _load_model_map() -> dict[str, dict[str, ModelInfo]]:
    model_map = collections.defaultdict(dict)
    config = load_config()
    for repo_name, repo_url in config.repos.items():
        server, owner, repo, branch = parse_repo_url(repo_url)
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


@app.command()
def list():
    pyaml.pprint(_load_model_map())


def get_serve_cmd(model: str):
    if ":" not in model:
        model = f"{model}:latest"
    bento_info = get_bento_info(model)
    if not bento_info:
        questionary.print(f"Model {model} not found", style=ERROR_STYLE)
        raise typer.Exit(1)
    cmd = ["bentoml", "serve", model]
    env = {
        "BENTOML_HOME": bento_info["model"]["repo"]["path"] + "/bentoml",
    }
    return cmd, env, None


def get_deploy_cmd(model: str):
    if ":" not in model:
        model = f"{model}:latest"
    bento_info = get_bento_info(model)
    if not bento_info:
        questionary.print(f"Model {model} not found", style=ERROR_STYLE)
        raise typer.Exit(1)

    cmd = ["bentoml", "deploy", model]
    env = {
        "BENTOML_HOME": bento_info["model"]["repo"]["path"] + "/bentoml",
    }

    required_envs = bento_info["bento_yaml"].get("envs", [])
    required_env_names = [env["name"] for env in required_envs if "name" in env]
    if required_env_names:
        questionary.print(
            f"This model requires the following environment variables to run: {repr(required_env_names)}",
            style="yellow",
        )

    for env_info in bento_info["bento_yaml"].get("envs", []):
        if "name" not in env_info:
            continue
        if os.environ.get(env_info["name"]):
            default = os.environ[env_info["name"]]
        elif "value" in env_info:
            default = env_info["value"]
        else:
            default = ""
        value = questionary.text(
            f"{env_info['name']}:",
            default=default,
        ).ask()
        if value is None:
            raise typer.Exit(1)
        cmd += ["--env", f"{env_info['name']}={value}"]
    return cmd, env, None


def get_bento_info(tag):
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


@app.command()
def get(tag: str):
    bento_info = get_bento_info(tag)
    if bento_info:
        pyaml.pprint(bento_info)
