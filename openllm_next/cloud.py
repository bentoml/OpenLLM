import asyncio
import json
import os
import subprocess

import questionary
import typer

from openllm_next.common import ERROR_STYLE, BentoInfo, run_command
from openllm_next.model import pick_bento

app = typer.Typer()


def _get_deploy_cmd(bento: BentoInfo):
    cmd = ["bentoml", "deploy", bento.tag]
    env = {
        "BENTOML_HOME": f"{bento.repo.path}/bentoml",
    }

    required_envs = bento.bento_yaml.get("envs", [])
    required_env_names = [env["name"] for env in required_envs if "name" in env]
    if required_env_names:
        questionary.print(
            f"This model requires the following environment variables to run: {repr(required_env_names)}",
            style="yellow",
        )

    for env_info in bento.bento_yaml.get("envs", []):
        if "name" not in env_info:
            continue
        if os.environ.get(env_info["name"]):
            default = os.environ[env_info["name"]]
        elif "value" in env_info:
            default = env_info["value"]
        else:
            default = ""
        value = questionary.text(
            f"{env_info['name']}:",
            default=default,
        ).ask()
        if value is None:
            raise typer.Exit(1)
        cmd += ["--env", f"{env_info['name']}={value}"]
    return cmd, env, None


def _ensure_cloud_context():
    cmd = ["bentoml", "cloud", "current-context"]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        context = json.loads(result)
        questionary.print(
            f"BentoCloud already logged in: {context['endpoint']}", style="green"
        )
    except subprocess.CalledProcessError:
        action = questionary.select(
            "BentoCloud not logged in",
            choices=[
                "I have a BentoCloud account",
                "get an account in two minutes",
            ],
        ).ask()
        if action is None:
            questionary.print("Cancelled", style=ERROR_STYLE)
            raise typer.Exit(1)
        elif action == "get an account in two minutes":
            questionary.print(
                "Please visit https://cloud.bentoml.com to get your token",
                style="yellow",
            )
        token = questionary.text("Enter your token: (similar to cniluaxxxxxxxx)").ask()
        if token is None:
            raise typer.Exit(1)
        endpoint = questionary.text(
            "Enter the endpoint: (similar to https://my-org.cloud.bentoml.com)"
        ).ask()
        if endpoint is None:
            raise typer.Exit(1)
        cmd = [
            "bentoml",
            "cloud",
            "login",
            "--api-token",
            token,
            "--endpoint",
            endpoint,
        ]
        try:
            result = subprocess.check_output(cmd)
            questionary.print("Logged in successfully", style="green")
        except subprocess.CalledProcessError:
            questionary.print("Failed to login", style=ERROR_STYLE)
            raise typer.Exit(1)


def serve(bento: BentoInfo):
    _ensure_cloud_context()
    cmd, env, cwd = _get_deploy_cmd(bento)
    run_command(cmd, env=env, cwd=cwd)


async def _run_model(bento: BentoInfo, timeout: int = 600):
    _ensure_cloud_context()
    cmd, env, cwd = _get_deploy_cmd(bento)
    server_proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    import bentoml
    from httpx import ReadError

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
            except ReadError:
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
    finally:
        questionary.print("\nStopping model server...", style="green")
        server_proc.terminate()
        questionary.print("Stopped model server", style="green")


def run(bento: BentoInfo):
    asyncio.run(_run_model(bento))
