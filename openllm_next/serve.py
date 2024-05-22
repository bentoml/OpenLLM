import asyncio
import os
import subprocess

import questionary
import typer

from openllm_next.common import run_command
from openllm_next.model import get_serve_cmd

app = typer.Typer()


@app.command()
def serve(model: str):
    cmd, env, cwd = get_serve_cmd(model)
    run_command(cmd, env=env, cwd=cwd)


async def _run_model(model: str, timeout: int = 600):
    cmd, env, cwd = get_serve_cmd(model)
    server_proc = subprocess.Popen(
        cmd,
        env={**os.environ, **env},
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
def run(model: str):
    asyncio.run(_run_model(model))
