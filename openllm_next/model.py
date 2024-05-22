import typer
import os

import questionary
import pyaml
from openllm_next.common import (
    ERROR_STYLE,
    load_config,
    BentoInfo,
)
from openllm_next.repo import parse_repo_url


app = typer.Typer()


def list_bento(tag: str | None = None) -> list[BentoInfo]:
    if tag is None:
        glob_pattern = "bentoml/bentos/*/*"
    elif ":" in tag:
        repo_name, version = tag.split(":")
        glob_pattern = f"bentoml/bentos/{repo_name}/{version}"
    else:
        glob_pattern = f"bentoml/bentos/{tag}/*"

    model_list = []
    config = load_config()
    for repo_name, repo_url in config.repos.items():
        repo = parse_repo_url(repo_url, repo_name)
        for path in repo.path.glob(glob_pattern):
            if path.is_dir() and (path / "bento.yaml").exists():
                model = BentoInfo(
                    repo=repo,
                    path=path,
                )
            elif path.is_file():
                with open(path) as f:
                    origin_name = f.read().strip()
                origin_path = path.parent / origin_name
                model = BentoInfo(
                    repo=repo,
                    path=origin_path,
                )
            else:
                model = None
            if model:
                model_list.append(model)
    return model_list


def pick_bento(tag) -> BentoInfo:
    model_list = list_bento(tag)
    if len(model_list) == 0:
        questionary.print("No models found", style=ERROR_STYLE)
        raise typer.Exit(1)
    if len(model_list) == 1:
        return model_list[0]
    model = questionary.select(
        "Select a model",
        choices=[questionary.Choice(str(model), value=model) for model in model_list],
    ).ask()
    if model is None:
        raise typer.Exit(1)
    return model


@app.command()
def list():
    pyaml.pprint(
        list_bento(),
        sort_dicts=False,
        sort_keys=False,
    )


def get_serve_cmd(tag: str):
    if ":" not in tag:
        tag = f"{tag}:latest"
    bento = pick_bento(tag)
    cmd = ["bentoml", "serve", bento.tag]
    env = {
        "BENTOML_HOME": f"{bento.repo.path}/bentoml",
    }
    return cmd, env, None


def get_deploy_cmd(tag: str):
    if ":" not in tag:
        tag = f"{tag}:latest"
    bento = pick_bento(tag)

    cmd = ["bentoml", "deploy", bento.tag]
    env = {
        "BENTOML_HOME": f"{bento.repo.path}/bentoml",
    }

    required_envs = bento.bento_yaml.get("envs", [])
    required_env_names = [env["name"] for env in required_envs if "name" in env]
    if required_env_names:
        questionary.print(
            f"This model requires the following environment variables to run: {repr(required_env_names)}",
            style="yellow",
        )

    for env_info in bento.bento_yaml.get("envs", []):
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


@app.command()
def get(tag: str):
    bento_info = pick_bento(tag)
    if bento_info:
        pyaml.pprint(
            bento_info.tolist(),
            sort_dicts=False,
            sort_keys=False,
        )
