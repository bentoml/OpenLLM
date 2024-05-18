import typer
from openllm_next.model import app as model_app
from openllm_next.repo import app as repo_app
from openllm_next.serve import serve as serve_serve


app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")


@app.command()
def serve(model: str):
    serve_serve(model)


if __name__ == "__main__":
    app()
