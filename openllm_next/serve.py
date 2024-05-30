import asyncio

import questionary
import typer

from openllm_next.accelerator_spec import match_machines
from openllm_next.common import run_command
from openllm_next.model import get_serve_cmd, list_bento, pick_bento
from openllm_next.venv import ensure_venv

app = typer.Typer()


@app.command()
def serve(model: str):
    if ":" not in model:
        model = f"{model}:latest"
    bento = pick_bento(model)
    venv = ensure_venv(bento)
    cmd, env, cwd = get_serve_cmd(bento)
    run_command(cmd, env=env, cwd=cwd, venv=venv)


async def _run_model(model: str, timeout: int = 600):
    if ":" not in model:
        model = f"{model}:latest"
    bento = pick_bento(model)
    venv = ensure_venv(bento)
    cmd, env, cwd = get_serve_cmd(bento)
    server_proc = run_command(
        cmd,
        env=env,
        cwd=cwd,
        venv=venv,
        silent=True,
        background=True,
    )

    import bentoml

    try:
        questionary.print("Model loading...", style="green")
        for _ in range(timeout):
            try:
                client = bentoml.AsyncHTTPClient(
                    "http://localhost:3000", timeout=timeout
                )
                resp = await client.request("GET", "/readyz")
                if resp.status_code == 200:
                    break
            except bentoml.exceptions.BentoMLException:
                await asyncio.sleep(1)
        else:
            questionary.print("Model failed to load", style="red")
            return

        questionary.print("Model is ready", style="green")
        messages = []
        while True:
            try:
                message = input("uesr: ")
                messages.append(dict(role="user", content=message))
                print("assistant: ", end="")
                assistant_message = ""
                async for text in client.chat(messages=messages):  # type: ignore
                    assistant_message += text
                    print(text, end="")
                messages.append(dict(role="assistant", content=assistant_message))
                print()

            except KeyboardInterrupt:
                break
    except asyncio.CancelledError:
        pass
    finally:
        questionary.print("\nStopping model server...", style="green")
        server_proc.terminate()
        questionary.print("Stopped model server", style="green")


@app.command()
def run(model: str = ""):
    if not model:
        models = list_bento()
        matchs = match_machines(
            [b.bento_yaml["services"][0]["config"]["resources"] for b in models]
        )
        selected = questionary.select(
            "Select a model to run",
            choices=[
                questionary.Choice(
                    f"{model.name}:{model.version} ({'local' if match[2] > 0 else 'cloud'})",
                    model,
                )
                for model, match in zip(models, matchs)
                if match[2] > 0
            ],
        ).ask()
        return
    asyncio.run(_run_model(model))
