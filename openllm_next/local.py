import asyncio
import time
import httpx

import questionary

from openllm_next.common import (
    BentoInfo,
    run_command,
    async_run_command,
    stream_command_output,
)
from openllm_next.venv import ensure_venv


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
    async with async_run_command(
        cmd,
        env=env,
        cwd=cwd,
        venv=venv,
        silent=False,
    ) as server_proc:
        import bentoml

        print("Model server started", server_proc.pid)

        stdout_streamer = None
        stderr_streamer = None
        start_time = time.time()

        questionary.print("Model loading...", style="green")
        for _ in range(timeout):
            try:
                resp = httpx.get("http://localhost:3000/readyz", timeout=3)
                if resp.status_code == 200:
                    break
            except httpx.RequestError:
                if time.time() - start_time > 30:
                    if not stdout_streamer:
                        stdout_streamer = asyncio.create_task(
                            stream_command_output(server_proc.stdout, style="gray")
                        )
                    if not stderr_streamer:
                        stderr_streamer = asyncio.create_task(
                            stream_command_output(server_proc.stderr, style="#BD2D0F")
                        )
                await asyncio.sleep(1)
        else:
            questionary.print("Model failed to load", style="red")
            server_proc.terminate()
            return

        if stdout_streamer:
            stdout_streamer.cancel()
        if stderr_streamer:
            stderr_streamer.cancel()

        questionary.print("Model is ready", style="green")
        messages = []
        client = bentoml.AsyncHTTPClient("http://localhost:3000", timeout=timeout)
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
        questionary.print("\nStopping model server...", style="green")
    questionary.print("Stopped model server", style="green")


def run(bento: BentoInfo):
    asyncio.run(_run_model(bento))
