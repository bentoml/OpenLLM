from typing import Annotated

import questionary
import typer

from openllm_next.accelerator_spec import match_deployment_target
from openllm_next.cloud import app as cloud_app
from openllm_next.common import VERBOSE_LEVEL
from openllm_next.local import run as local_run
from openllm_next.local import serve as local_serve
from openllm_next.model import app as model_app
from openllm_next.model import list_bento
from openllm_next.repo import app as repo_app

app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")
app.add_typer(cloud_app, name="cloud")


def _pre_select(model: str):
    bentos = list_bento(model)
    if len(bentos) == 0:
        typer.echo(f"No model found for {model}", err=True)
        raise typer.Exit(1)
    matchs = match_deployment_target(bentos)
    if len(bentos) == 1:
        bento = bentos[0]
    else:
        choices = []
        # bentos that can be run locally
        choices += [questionary.Separator("Local available models")]
        choices += [
            questionary.Choice(
                f"  {bento.name}:{bento.version}",
                bento,
            )
            for bento, targets in matchs.items()
            if targets
        ]
        choices += [questionary.Separator("Cloud available models")]
        choices += [
            questionary.Choice(
                f"  {bento.name}:{bento.version}",
                bento,
            )
            for bento, targets in matchs.items()
            if not targets
        ]

        bento = questionary.select(
            "Select a model to run",
            choices=choices,
        ).ask()
        if not bento:
            questionary.print("No model selected", style="red")
            raise typer.Exit(1)
    return bento, matchs[bento]


@app.command()
def serve(model: str):
    bento, targets = _pre_select(model)
    local_serve(bento)


@app.command()
def run(model: Annotated[str, typer.Argument()] = ""):
    bento, targets = _pre_select(model)
    if targets:
        local_run(bento)
    else:
        typer.echo("No local available deployment target found", err=True)
        raise typer.Exit(1)


def typer_callback(verbose: int = 0):
    if verbose:
        VERBOSE_LEVEL.set(verbose)


def main():
    app.callback()(typer_callback)
    app()


if __name__ == "__main__":
    main()
