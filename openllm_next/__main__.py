import typer
from openllm_next.model import app as model_app
from openllm_next.repo import app as repo_app
from openllm_next.serve import serve as local_serve, run as local_run


app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")


@app.command()
def serve(model: str):
    local_serve(model)


@app.command()
def run(model: str):
    local_run(model)


if __name__ == "__main__":
    app()
