import json
import os
import pathlib
import shutil
import subprocess

import typing
import questionary
import typer

from openllm_next.accelerator_spec import ACCELERATOR_SPECS
from openllm_next.common import (
    ERROR_STYLE,
    BentoInfo,
    DeploymentTarget,
    output,
    run_command,
)

app = typer.Typer()


def _get_deploy_cmd(bento: BentoInfo, target: typing.Optional[DeploymentTarget] = None):
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

    if target:
        cmd += ["--instance-type", target.name]

    assert (pathlib.Path.home() / "bentoml" / ".yatai.yaml").exists()
    shutil.copy(
        pathlib.Path.home() / "bentoml" / ".yatai.yaml",
        bento.repo.path / "bentoml" / ".yatai.yaml",
    )

    return cmd, env, None


def ensure_cloud_context():
    cmd = ["bentoml", "cloud", "current-context"]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        context = json.loads(result)
        output(f"  BentoCloud already logged in: {context['endpoint']}", style="green")
    except subprocess.CalledProcessError:
        action = questionary.select(
            "BentoCloud not logged in",
            choices=[
                "I have a BentoCloud account",
                "get an account in two minutes",
            ],
        ).ask()
        if action is None:
            raise typer.Exit(1)
        elif action == "get an account in two minutes":
            output(
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
            output("  Logged in successfully", style="green")
        except subprocess.CalledProcessError:
            output("  Failed to login", style=ERROR_STYLE)
            raise typer.Exit(1)


def get_cloud_machine_spec():
    cmd = ["bentoml", "deployment", "list-instance-types", "-o", "json"]
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        instance_types = json.loads(result)
        return [
            DeploymentTarget(
                source="cloud",
                name=it["name"],
                price=it["price"],
                platform="linux",
                accelerators=(
                    [ACCELERATOR_SPECS[it["gpu_type"]] for _ in range(int(it["gpu"]))]
                    if it.get("gpu") and it["gpu_type"] in ACCELERATOR_SPECS
                    else []
                ),
            )
            for it in instance_types
        ]
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        questionary.print("Failed to get cloud instance types", style=ERROR_STYLE)
        return []


def serve(bento: BentoInfo, target: DeploymentTarget):
    ensure_cloud_context()
    cmd, env, cwd = _get_deploy_cmd(bento, target)
    run_command(cmd, env=env, cwd=cwd)
