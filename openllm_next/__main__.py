from typing import Annotated, Optional
from collections import defaultdict
import sys
import questionary

import typer

from openllm_next.accelerator_spec import (
    DeploymentTarget,
    can_run,
    get_local_machine_spec,
)
from openllm_next.cloud import app as cloud_app, ensure_cloud_context
from openllm_next.cloud import get_cloud_machine_spec
from openllm_next.cloud import serve as cloud_deploy
from openllm_next.common import VERBOSE_LEVEL, BentoInfo, FORCE, output
from openllm_next.local import run as local_run
from openllm_next.local import serve as local_serve
from openllm_next.model import app as model_app
from openllm_next.model import list_bento
from openllm_next.repo import app as repo_app

app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")
app.add_typer(cloud_app, name="cloud")


def _pick_bento(model: str, target: Optional[DeploymentTarget] = None) -> BentoInfo:
    bentos = list_bento(model)
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
            output(f"  {bento}", level=20, style="red")
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
            output(f"  {bento}", level=20, style="red")
        raise typer.Exit(1)

    return bentos[0]


def _select_bento_name(models, target):
    from tabulate import tabulate

    options = []
    model_infos = [
        [model.repo.name, model.name, model.tag, can_run(model, target)]
        for model in models
    ]
    model_name_groups = defaultdict(lambda: 0)
    for repo, name, tag, score in model_infos:
        model_name_groups[(repo, name)] += score
    table_data = [
        [name, repo, "*" if score > 0 else ""]
        for (repo, name), score in model_name_groups.items()
    ]
    table = tabulate(
        table_data,
        headers=["model", "repo", "locally runnable"],
    ).split("\n")
    headers = f"{table[0]}\n   {table[1]}"

    options.append(questionary.Separator(headers))
    for table_data, table_line in zip(table_data, table[2:]):
        options.append(questionary.Choice(table_line, value=table_data[:2]))
    selected = questionary.select("Select a model", options).ask()
    if selected is None:
        raise typer.Exit(1)
    return selected


def _select_bento_version(models, target, bento_name, repo):
    from tabulate import tabulate

    model_infos = [
        [model, can_run(model, target)]
        for model in models
        if model.name == bento_name and model.repo.name == repo
    ]

    table_data = [
        [model.version, "yes" if score > 0 else ""]
        for model, score in model_infos
        if model.name == bento_name and model.repo.name == repo
    ]
    if not table_data:
        output(f"No model found for {bento_name} in {repo}", level=20, style="red")
        raise typer.Exit(1)
    table = tabulate(
        table_data,
        headers=["version", "locally runnable"],
    ).split("\n")

    options = []
    options.append(questionary.Separator(f"{table[0]}\n   {table[1]}"))
    for table_data, table_line in zip(model_infos, table[2:]):
        options.append(questionary.Choice(table_line, value=table_data))
    selected = questionary.select("Select a version", options).ask()
    if selected is None:
        raise typer.Exit(1)
    return selected


def _select_target(bento, targets):
    from tabulate import tabulate

    options = []
    targets.sort(key=lambda x: can_run(bento, x), reverse=True)
    if not targets:
        output(
            "No available instance type, check your bentocloud account",
            level=20,
            style="red",
        )
        raise typer.Exit(1)

    table = tabulate(
        [
            [
                target.name,
                target.accelerators_repr,
                target.price,
                "" if can_run(bento, target) else "insufficient res.",
            ]
            for target in targets
        ],
        headers=["instance type", "accelerator", "price", "deployable"],
    ).split("\n")
    options.append(questionary.Separator(f"{table[0]}\n   {table[1]}"))

    for target, line in zip(targets, table[2:]):
        options.append(
            questionary.Choice(
                f"{line}",
                value=target,
            )
        )
    selected = questionary.select("Select an instance type", options).ask()
    if selected is None:
        raise typer.Exit(1)
    return selected


def _select_action(bento, score):
    if score > 0:
        options = [
            questionary.Separator("Available actions"),
            questionary.Separator("0. Run the model in terminal"),
            questionary.Choice(f"  $ openllm run {bento}", value="run"),
            questionary.Separator(" "),
            questionary.Separator("1. Serve the model locally and get a chat server"),
            questionary.Choice(f"  $ openllm serve {bento}", value="serve"),
            questionary.Separator(" "),
            questionary.Separator(
                "2. Deploy the model to bentocloud and get a scalable chat server"
            ),
            questionary.Choice(f"  $ openllm deploy {bento}", value="deploy"),
        ]
    else:
        options = [
            questionary.Separator("Available actions"),
            questionary.Separator("0. Run the model in terminal"),
            questionary.Choice(
                f"  $ openllm run {bento}",
                value="run",
                disabled="insufficient resources",
                shortcut_key="0",
            ),
            questionary.Separator(" "),
            questionary.Separator("1. Serve the model locally and get a chat server"),
            questionary.Choice(
                f"  $ openllm serve {bento}",
                value="serve",
                disabled="insufficient resources",
                shortcut_key="1",
            ),
            questionary.Separator(" "),
            questionary.Separator(
                "2. Deploy the model to bentocloud and get a scalable chat server"
            ),
            questionary.Choice(
                f"  $ openllm deploy {bento}",
                value="deploy",
                shortcut_key="2",
            ),
        ]
    action = questionary.select("Select an action", options).ask()
    if action is None:
        raise typer.Exit(1)
    if action == "run":
        local_run(bento)
    elif action == "serve":
        local_serve(bento)
    elif action == "deploy":
        ensure_cloud_context()
        targets = get_cloud_machine_spec()
        target = _select_target(bento, targets)
        cloud_deploy(bento, target)


@app.command()
def hello():
    target = get_local_machine_spec()
    output(f"  Detected Platform: {target.platform}", style="green")
    if target.accelerators:
        output("  Detected Accelerators: ", style="green")
        for a in target.accelerators:
            output(f"   - {a.model} {a.memory_size}GB", style="green")
    else:
        output("  Detected Accelerators: None", style="yellow")

    models = list_bento()

    bento_name, repo = _select_bento_name(models, target)
    bento, score = _select_bento_version(models, target, bento_name, repo)
    _select_action(bento, score)


@app.command()
def serve(model: Annotated[str, typer.Argument()] = ""):
    target = get_local_machine_spec()
    bento = _pick_bento(model, target)
    local_serve(bento)


@app.command()
def run(model: Annotated[str, typer.Argument()] = ""):
    target = get_local_machine_spec()
    bento = _pick_bento(model, target)
    local_run(bento)


@app.command()
def deploy(model: Annotated[str, typer.Argument()] = ""):
    targets = get_cloud_machine_spec()


def typer_callback(verbose: int = 0):
    if verbose:
        VERBOSE_LEVEL.set(verbose)


def main():
    if sys.version_info < (3, 9):
        output("Python 3.8 or higher is required", level=20, style="red")
        sys.exit(1)
    app.callback()(typer_callback)
    app()


if __name__ == "__main__":
    main()
