import collections
import typing
from typing import Optional

import pyaml
import typer

from openllm_next.common import (
    VERBOSE_LEVEL,
    BentoInfo,
    load_config,
    output,
    FORCE,
)
from openllm_next.repo import ensure_repo_updated, parse_repo_url
from openllm_next.accelerator_spec import can_run, DeploymentTarget


app = typer.Typer()


@app.command()
def get(tag: str, repo: Optional[str] = None):
    bento_info = ensure_bento(tag, repo_name=repo)
    if bento_info:
        with VERBOSE_LEVEL.patch(1):
            pyaml.pprint(
                bento_info,
                sort_dicts=False,
                sort_keys=False,
            )


@app.command(name="list")
def list_(repo: Optional[str] = None):
    bentos = list_bento(repo_name=repo)
    output: dict[str, list[str]] = collections.defaultdict(list)
    for bento in bentos:
        output[bento.name].append(bento.version)
    pyaml.pprint(
        output,
        sort_dicts=False,
        sort_keys=False,
    )


def ensure_bento(
    model: str,
    target: Optional[DeploymentTarget] = None,
    repo_name: Optional[str] = None,
) -> BentoInfo:
    bentos = list_bento(model, repo_name=repo_name)
    if len(bentos) == 0:
        output(f"No model found for {model}", level=20, style="red")
        raise typer.Exit(1)

    if len(bentos) == 1:
        if FORCE.get():
            output(f"Found model {bentos[0]}", level=10, style="green")
            return bentos[0]
        if target is None:
            return bentos[0]
        if can_run(bentos[0], target) <= 0:
            return bentos[0]
        output(f"Found model {bentos[0]}", level=10, style="green")
        return bentos[0]

    if target is None:
        output(
            f"Multiple models match {model}, did you mean one of these?",
            level=20,
            style="red",
        )
        for bento in bentos:
            output(f"  {bento}", level=20)
        raise typer.Exit(1)

    filtered = [bento for bento in bentos if can_run(bento, target) > 0]
    if len(filtered) == 0:
        output(f"No deployment target found for {model}", level=20, style="red")
        raise typer.Exit(1)

    if len(filtered) == 0:
        output(f"No deployment target found for {model}", level=20, style="red")
        raise typer.Exit(1)

    if len(bentos) > 1:
        output(
            f"Multiple models match {model}, did you mean one of these?",
            level=20,
            style="red",
        )
        for bento in bentos:
            output(f"  {bento}", level=20)
        raise typer.Exit(1)

    return bentos[0]


def list_bento(
    tag: typing.Optional[str] = None,
    repo_name: typing.Optional[str] = None,
    include_alias: bool = False,
) -> typing.List[BentoInfo]:
    ensure_repo_updated()

    if repo_name is not None:
        config = load_config()
        if repo_name not in config.repos:
            output(
                f"Repo `{repo_name}` not found, did you mean one of these?", level=20
            )
            for repo_name in config.repos:
                output(f"  {repo_name}", level=20)
            raise typer.Exit(1)

    if not tag:
        glob_pattern = "bentoml/bentos/*/*"
    elif ":" in tag:
        bento_name, version = tag.split(":")
        glob_pattern = f"bentoml/bentos/{bento_name}/{version}"
    else:
        glob_pattern = f"bentoml/bentos/{tag}/*"

    model_list = []
    config = load_config()
    for _repo_name, repo_url in config.repos.items():
        if repo_name is not None and _repo_name != repo_name:
            continue
        repo = parse_repo_url(repo_url, _repo_name)
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
    if not include_alias:
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
