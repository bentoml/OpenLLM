from typing import Annotated

import questionary
import typer

from openllm_next.accelerator_spec import (
    DeploymentTarget,
    can_run,
    get_local_machine_spec,
)
from openllm_next.cloud import app as cloud_app
from openllm_next.cloud import get_cloud_machine_spec
from openllm_next.cloud import run as cloud_run
from openllm_next.cloud import serve as cloud_serve
from openllm_next.common import VERBOSE_LEVEL, BentoInfo
from openllm_next.local import run as local_run
from openllm_next.local import serve as local_serve
from openllm_next.model import app as model_app
from openllm_next.model import list_bento
from openllm_next.repo import app as repo_app

app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")
app.add_typer(cloud_app, name="cloud")


def _pre_select(model: str) -> tuple[BentoInfo, DeploymentTarget]:
    bentos = list_bento(model)
    if len(bentos) == 0:
        typer.echo(f"No model found for {model}", err=True)
        raise typer.Exit(1)

    local = get_local_machine_spec()

    if len(bentos) == 1:
        bento = bentos[0]
        if can_run(bento, local) <= 0:
            questionary.print(
                f"No deployment target found for {bento.name}:{bento.version}",
                style="red",
            )
            raise typer.Exit(1)
        return bento, local

    choices = []
    choices += [questionary.Separator("Local available models")]
    choices += [
        questionary.Choice(
            f"  {bento.name}:{bento.version}",
            (bento, local),
        )
        for bento in bentos
        if can_run(bento) > 0
    ]
    choices += [questionary.Separator("Cloud available models")]
    choices += [
        questionary.Choice(
            f"  {bento.name}:{bento.version}",
            (bento, None),
        )
        for bento in bentos
    ]

    choosen: tuple[BentoInfo, DeploymentTarget] = questionary.select(
        "Select a model to run",
        choices=choices,
    ).ask()

    if not choosen:
        questionary.print("No model selected", style="red")
        raise typer.Exit(1)

    bento, target = choosen
    if target is None:
        cloud_targets = get_cloud_machine_spec()
        cloud_targets = [
            target for target in cloud_targets if can_run(bento, target) > 0
        ]
        if len(cloud_targets) == 0:
            questionary.print(
                f"No suitable instance type found for {bento.name}:{bento.version}",
                style="red",
            )
            raise typer.Exit(1)
        target = questionary.select(
            "Select a cloud target",
            choices=[
                questionary.Choice(
                    f"  {target.name}",
                    target,
                )
                for target in cloud_targets
            ],
        ).ask()
        if not target:
            questionary.print("No target selected", style="red")
            raise typer.Exit(1)

    return bento, target


@app.command()
def serve(model: Annotated[str, typer.Argument()] = ""):
    bento, target = _pre_select(model)
    if target and target.source == "local":
        local_serve(bento)
    else:
        cloud_serve(bento, target)


@app.command()
def run(model: Annotated[str, typer.Argument()] = ""):
    bento, target = _pre_select(model)
    if target and target.source == "local":
        local_run(bento)
    else:
        cloud_run(bento, target)


def typer_callback(verbose: int = 0):
    if verbose:
        VERBOSE_LEVEL.set(verbose)


def main():
    app.callback()(typer_callback)
    app()


if __name__ == "__main__":
    main()
