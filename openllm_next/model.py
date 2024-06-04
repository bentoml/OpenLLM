import collections
import typing

import pyaml
import questionary
import typer

from openllm_next.common import (
    ERROR_STYLE,
    VERBOSE_LEVEL,
    BentoInfo,
    load_config,
)
from openllm_next.repo import ensure_repo_updated, parse_repo_url

app = typer.Typer()


@app.command()
def get(tag: str):
    bento_info = pick_bento(tag)
    if bento_info:
        with VERBOSE_LEVEL.patch(1):
            pyaml.pprint(
                bento_info,
                sort_dicts=False,
                sort_keys=False,
            )


@app.command(name="list")
def list_():
    bentos = list_bento()
    output: dict[str, list[str]] = collections.defaultdict(list)
    for bento in bentos:
        output[bento.name].append(bento.version)
    pyaml.pprint(
        output,
        sort_dicts=False,
        sort_keys=False,
    )


def list_bento(tag: typing.Optional[str] = None) -> typing.List[BentoInfo]:
    ensure_repo_updated()
    if not tag:
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
    model_list.sort(key=lambda x: x.tag)
    if VERBOSE_LEVEL.get() <= 0:
        seen = set()
        model_list = [
            x
            for x in model_list
            if not (
                f"{x.bento_yaml['name']}:{x.bento_yaml['version']}" in seen
                or seen.add(f"{x.bento_yaml['name']}:{x.bento_yaml['version']}")
            )
        ]
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
