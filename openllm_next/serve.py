import typer
import questionary
from openllm_next.common import ERROR_STYLE, run_command
from openllm_next.model import _get_bento_info

app = typer.Typer()


def _serve_model(model: str):
    if ":" not in model:
        model = f"{model}:latest"
    bento_info = _get_bento_info(model)
    if not bento_info:
        questionary.print(f"Model {model} not found", style=ERROR_STYLE)
        return
    cmd = ["bentoml", "serve", model]
    env = {
        "CLLAMA_MODEL": model,
        "BENTOML_HOME": bento_info["model"]["repo"]["path"] + "/bentoml",
    }
    run_command(cmd, env=env)


@app.command()
def serve(model: str):
    _serve_model(model)
