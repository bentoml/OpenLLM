import re
import typing
from typing import Optional

import tabulate
import typer

from openllm.accelerator_spec import DeploymentTarget, can_run
from openllm.analytic import OpenLLMTyper
from openllm.common import FORCE, VERBOSE_LEVEL, BentoInfo, load_config, output
from openllm.repo import ensure_repo_updated, parse_repo_url

app = OpenLLMTyper(help='manage models')


@app.command(help='get model')
def get(tag: str, repo: Optional[str] = None, verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    bento_info = ensure_bento(tag, repo_name=repo)
    if bento_info:
        output(bento_info)


@app.command(name='list', help='list available models')
def list_model(tag: Optional[str] = None, repo: Optional[str] = None, verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)

    bentos = list_bento(tag=tag, repo_name=repo)
    bentos.sort(key=lambda x: x.name)

    seen = set()

    def is_seen(value):
        if value in seen:
            return True
        seen.add(value)
        return False

    table = tabulate.tabulate(
        [
            [
                '' if is_seen(bento.name) else bento.name,
                bento.tag,
                bento.repo.name,
                bento.pretty_gpu,
                ','.join(bento.platforms),
            ]
            for bento in bentos
        ],
        headers=['model', 'version', 'repo', 'required GPU RAM', 'platforms'],
    )
    output(table)


def ensure_bento(model: str, target: Optional[DeploymentTarget] = None, repo_name: Optional[str] = None) -> BentoInfo:
    bentos = list_bento(model, repo_name=repo_name)
    if len(bentos) == 0:
        output(f'No model found for {model}', style='red')
        raise typer.Exit(1)

    if len(bentos) == 1:
        if FORCE.get():
            output(f'Found model {bentos[0]}', style='green')
            return bentos[0]
        if target is None:
            return bentos[0]
        if can_run(bentos[0], target) <= 0:
            return bentos[0]
        output(f'Found model {bentos[0]}', style='green')
        return bentos[0]

    if target is None:
        output(f'Multiple models match {model}, did you mean one of these?', style='red')
        for bento in bentos:
            output(f'  {bento}')
        raise typer.Exit(1)

    filtered = [bento for bento in bentos if can_run(bento, target) > 0]
    if len(filtered) == 0:
        output(f'No deployment target found for {model}', style='red')
        raise typer.Exit(1)

    if len(filtered) == 0:
        output(f'No deployment target found for {model}', style='red')
        raise typer.Exit(1)

    if len(bentos) > 1:
        output(f'Multiple models match {model}, did you mean one of these?', style='red')
        for bento in bentos:
            output(f'  {bento}')
        raise typer.Exit(1)

    return bentos[0]


NUMBER_RE = re.compile(r'\d+')


def _extract_first_number(s: str):
    match = NUMBER_RE.search(s)
    if match:
        return int(match.group())
    else:
        return 100


def list_bento(
    tag: typing.Optional[str] = None, repo_name: typing.Optional[str] = None, include_alias: bool = False
) -> typing.List[BentoInfo]:
    ensure_repo_updated()

    if repo_name is not None:
        config = load_config()
        if repo_name not in config.repos:
            output(f'Repo `{repo_name}` not found, did you mean one of these?')
            for repo_name in config.repos:
                output(f'  {repo_name}')
            raise typer.Exit(1)

    if not tag:
        glob_pattern = 'bentoml/bentos/*/*'
    elif ':' in tag:
        bento_name, version = tag.split(':')
        glob_pattern = f'bentoml/bentos/{bento_name}/{version}'
    else:
        glob_pattern = f'bentoml/bentos/{tag}/*'

    model_list = []
    config = load_config()
    for _repo_name, repo_url in config.repos.items():
        if repo_name is not None and _repo_name != repo_name:
            continue
        repo = parse_repo_url(repo_url, _repo_name)

        paths = sorted(
            repo.path.glob(glob_pattern),
            key=lambda x: (x.parent.name, _extract_first_number(x.name), len(x.name), x.name),
        )
        for path in paths:
            if path.is_dir() and (path / 'bento.yaml').exists():
                model = BentoInfo(repo=repo, path=path)
            elif path.is_file():
                with open(path) as f:
                    origin_name = f.read().strip()
                origin_path = path.parent / origin_name
                model = BentoInfo(alias=path.name, repo=repo, path=origin_path)
            else:
                model = None
            if model:
                model_list.append(model)

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
