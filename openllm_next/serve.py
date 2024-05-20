import typer
import asyncio
import questionary
from openllm_next.common import ERROR_STYLE, run_command
from openllm_next.model import _get_bento_info

app = typer.Typer()


def _serve_model(model: str, bg: bool = False):
    if ":" not in model:
        model = f"{model}:latest"
    bento_info = _get_bento_info(model)
    if not bento_info:
        questionary.print(f"Model {model} not found", style=ERROR_STYLE)
        return
    cmd = ["bentoml", "serve", model]
    env = {
        "BENTOML_HOME": bento_info["model"]["repo"]["path"] + "/bentoml",
    }
    return run_command(cmd, env=env, bg=bg)


@app.command()
def serve(model: str):
    _serve_model(model)


async def _run_model(model: str, timeout: int = 600):
    server_proc = _serve_model(model, bg=True)
    assert server_proc is not None

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
    finally:
        questionary.print("\nStopping model server...", style="green")
        server_proc.terminate()
        questionary.print("Stopped model server", style="green")


@app.command()
def run(model: str):
    asyncio.run(_run_model(model))
