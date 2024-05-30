import asyncio

import questionary
import typer

from openllm_next.common import BentoInfo, run_command
from openllm_next.venv import ensure_venv

app = typer.Typer()


def _get_serve_cmd(bento: BentoInfo):
    cmd = ["bentoml", "serve", bento.tag]
    env = {
        "BENTOML_HOME": f"{bento.repo.path}/bentoml",
    }
    return cmd, env, None


def serve(bento: BentoInfo):
    venv = ensure_venv(bento)
    cmd, env, cwd = _get_serve_cmd(bento)
    run_command(cmd, env=env, cwd=cwd, venv=venv)


async def _run_model(bento: BentoInfo, timeout: int = 600):
    venv = ensure_venv(bento)
    cmd, env, cwd = _get_serve_cmd(bento)
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
                message = input("user: ")
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


def run(bento: BentoInfo):
    asyncio.run(_run_model(bento))
