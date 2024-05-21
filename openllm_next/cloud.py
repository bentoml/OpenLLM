import typer
import json
import subprocess
import asyncio
import questionary
from openllm_next.common import ERROR_STYLE, run_command
from openllm_next.model import get_bento_info, get_deploy_cmd

app = typer.Typer()


def _ensure_cloud_context():
    cmd = ["bentoml", "cloud", "current-context"]
    try:
        result = subprocess.check_output(cmd)
        context = json.loads(result)
        questionary.print(f"already logged in to {context['endpoint']}", style="green")
    except subprocess.CalledProcessError:
        action = questionary.select(
            "bento cloud not logged in",
            choices=[
                "I have a token",
                "Get a token in two minutes",
            ],
        ).ask()
        if action is None:
            questionary.print("Cancelled", style=ERROR_STYLE)
            raise typer.Exit(1)
        elif action == "Get a token in two minutes":
            questionary.print(
                "Please visit https://cloud.bentoml.com to get your token",
                style="green",
            )
        token = questionary.text("Enter your token: (like cniluaxxxxxxxx)").ask()
        if token is None:
            questionary.print("Cancelled", style=ERROR_STYLE)
            raise typer.Exit(1)
        endpoint = questionary.text(
            "Enter the endpoint: (like https://my-org.cloud.bentoml.com)"
        ).ask()
        if endpoint is None:
            questionary.print("Cancelled", style=ERROR_STYLE)
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


@app.command()
def serve(model: str):
    _ensure_cloud_context()
    cmd, env, cwd = get_deploy_cmd(model)
    run_command(cmd, env=env, cwd=cwd)


async def _run_model(model: str, timeout: int = 600):
    _ensure_cloud_context()
    cmd, env, cwd = get_deploy_cmd(model)
    server_proc = subprocess.Popen(
        cmd,
        env=env,
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
    finally:
        questionary.print("\nStopping model server...", style="green")
        server_proc.terminate()
        questionary.print("Stopped model server", style="green")


@app.command()
def run(model: str):
    asyncio.run(_run_model(model))
