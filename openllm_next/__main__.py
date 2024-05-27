import typer

from openllm_next.cloud import app as cloud_app
from openllm_next.common import VERBOSE_LEVEL
from openllm_next.model import app as model_app
from openllm_next.repo import app as repo_app
from openllm_next.serve import run as local_run
from openllm_next.serve import serve as local_serve

app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")
app.add_typer(cloud_app, name="cloud")


@app.command()
def serve(model: str):
    local_serve(model)


@app.command()
def run(model: str):
    local_run(model)


def typer_callback(verbose: int = 0):
    if verbose:
        VERBOSE_LEVEL.set(verbose)


def main():
    app.callback()(typer_callback)
    app()


if __name__ == "__main__":
    main()
