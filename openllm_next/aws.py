"""
Deprecated
"""

import typer
import typing
import collections

import prompt_toolkit
from prompt_toolkit import print_formatted_text as print
import time
import uuid
import shutil
import pydantic
from urllib.parse import urlparse
import yaml
import json
import bentoml
import questionary
import os
import re
import subprocess
import pyaml
import pathlib
from cllama.spec import GPU_MEMORY

ERROR_STYLE = "red"
SUCCESS_STYLE = "green"


CLLAMA_HOME = pathlib.Path.home() / ".openllm_next"
REPO_DIR = CLLAMA_HOME / "repos"
TEMP_DIR = CLLAMA_HOME / "temp"
VENV_DIR = CLLAMA_HOME / "venv"

REPO_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR.mkdir(exist_ok=True, parents=True)
VENV_DIR.mkdir(exist_ok=True, parents=True)

CONFIG_FILE = CLLAMA_HOME / "config.json"


app = typer.Typer()
repo_app = typer.Typer()
model_app = typer.Typer()

app.add_typer(repo_app, name="repo")
app.add_typer(model_app, name="model")


class Config(pydantic.BaseModel):
    repos: dict[str, str] = {
        "default": "git+https://github.com/bojiang/bentovllm@main#subdirectory=bentoml"
    }
    default_repo: str = "default"


def _load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return Config(**json.load(f))
    return Config()


def _save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.dict(), f, indent=2)


class ModelInfo(typing.TypedDict):
    repo: str
    path: str


def _load_model_map() -> dict[str, dict[str, ModelInfo]]:
    model_map = collections.defaultdict(dict)
    config = _load_config()
    for repo_name, repo_url in config.repos.items():
        server, owner, repo, _ = _parse_repo_url(repo_url)
        repo_dir = REPO_DIR / server / owner / repo
        for path in repo_dir.glob("bentoml/bentos/*/*"):
            if path.is_dir():
                model_map[path.parent.name][path.name] = ModelInfo(
                    repo=repo_name,
                    path=str(path),
                )
            elif path.is_file():
                with open(path) as f:
                    origin_name = f.read().strip()
                origin_path = path.parent / origin_name
                model_map[path.parent.name][path.name] = ModelInfo(
                    repo=repo_name,
                    path=str(origin_path),
                )
    return model_map


GIT_REPO_RE = re.compile(
    r"git\+https://(?P<server>.+)/(?P<owner>.+)/(?P<repo>.+?)(@(?P<branch>.+))?$"
)


@repo_app.command(name="list")
def repo_list():
    config = _load_config()
    pyaml.pprint(config.repos)


def _parse_repo_url(repo_url):
    """
    parse the git repo url to server, owner, repo name, branch
    >>> _parse_repo_url("git+https://github.com/bojiang/bentovllm@main")
    ('github.com', 'bojiang', 'bentovllm', 'main')

    >>> _parse_repo_url("git+https://github.com/bojiang/bentovllm")
    ('github.com', 'bojiang', 'bentovllm', 'main')
    """
    match = GIT_REPO_RE.match(repo_url)
    if not match:
        raise ValueError(f"Invalid git repo url: {repo_url}")
    return (
        match.group("server"),
        match.group("owner"),
        match.group("repo"),
        match.group("branch") or "main",
    )


@repo_app.command(name="add")
def repo_add(name: str, repo: str):
    name = name.lower()
    if not name.isidentifier():
        questionary.print(
            f"Invalid repo name: {name}, should only contain letters, numbers and underscores",
            style=ERROR_STYLE,
        )
        return

    config = _load_config()
    if name in config.repos:
        override = questionary.confirm(
            f"Repo {name} already exists({config.repos[name]}), override?"
        ).ask()
        if not override:
            return

    config.repos[name] = repo
    _save_config(config)
    pyaml.pprint(config.repos)


@repo_app.command(name="remove")
def repo_remove(name: str):
    config = _load_config()
    if name not in config.repos:
        questionary.print(f"Repo {name} does not exist", style=ERROR_STYLE)
        return

    del config.repos[name]
    _save_config(config)
    pyaml.pprint(config.repos)


def _run_command(cmd, cwd=None):
    questionary.print(f"\n$ {' '.join(cmd)}", style="bold")
    subprocess.run(cmd, cwd=cwd, check=True)


@repo_app.command(name="update")
def repo_update():
    config = _load_config()
    repos_in_use = set()
    for name, repo in config.repos.items():
        server, owner, repo_name, branch = _parse_repo_url(repo)
        repos_in_use.add((server, owner, repo_name))
        repo_dir = REPO_DIR / server / owner / repo_name
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                cmd = [
                    "git",
                    "clone",
                    "--branch",
                    branch,
                    f"https://{server}/{owner}/{repo_name}.git",
                    str(repo_dir),
                ]
                _run_command(cmd)
            except subprocess.CalledProcessError:
                shutil.rmtree(repo_dir, ignore_errors=True)
                questionary.print(f"Failed to clone repo {name}", style=ERROR_STYLE)
        else:
            try:
                cmd = ["git", "fetch", "origin", branch]
                _run_command(cmd, cwd=repo_dir)
                cmd = ["git", "reset", "--hard", f"origin/{branch}"]
                _run_command(cmd, cwd=repo_dir)
            except:
                shutil.rmtree(repo_dir, ignore_errors=True)
                questionary.print(f"Failed to update repo {name}", style=ERROR_STYLE)
    for repo_dir in REPO_DIR.glob("*/*/*"):
        if tuple(repo_dir.parts[-3:]) not in repos_in_use:
            shutil.rmtree(repo_dir, ignore_errors=True)
            questionary.print(f"Removed unused repo {repo_dir}")
    questionary.print("Repos updated", style=SUCCESS_STYLE)


@model_app.command(name="list")
def model_list():
    pyaml.pprint(_load_model_map())


def get_bento_info(tag):
    model_map = _load_model_map()
    bento, version = tag.split(":")
    if bento not in model_map or version not in model_map[bento]:
        questionary.print(f"Model {tag} not found", style=ERROR_STYLE)
        return
    model_info = model_map[bento][version]
    repo_name = model_info["repo"]
    path = pathlib.Path(model_info["path"])

    bento_file = path / "bento.yaml"
    bento_info = yaml.safe_load(bento_file.read_text())
    return bento_info


@model_app.command(name="get")
def model_get(tag: str):
    bento_info = get_bento_info(tag)
    if bento_info:
        pyaml.pprint(bento_info)


def _filter_instance_types(
    instance_types,
    gpu_count,
    gpu_memory=None,
    gpu_type=None,
    level="match",
):
    if gpu_memory is None:
        if gpu_type is None:
            raise ValueError("Either gpu_memory or gpu_type must be provided")
        gpu_memory = GPU_MEMORY[gpu_type]

    def _check_instance(spec):
        if gpu_count == 0 or gpu_count is None:
            if "GpuInfo" in spec:
                return False
            else:
                return True
        else:
            gpus = spec.get("GpuInfo", {}).get("Gpus", [])
            if len(gpus) == 0:
                return False
            it_gpu = gpus[0]
            it_gpu_mem = it_gpu["MemoryInfo"]["SizeInMiB"] / 1024

            if it_gpu["Count"] == gpu_count and it_gpu_mem == gpu_memory:
                return True
            elif it_gpu["Count"] >= gpu_count and it_gpu_mem >= gpu_memory:
                if level == "match":
                    return False
                elif level == "usable":
                    return True
                else:
                    assert False
            else:
                return False

    def _sort_key(spec):
        return (
            spec["InstanceType"].split(".")[0],
            spec.get("GpuInfo", {}).get("Gpus", [{}])[0].get("Count", 0),
            spec.get("VCpuInfo", {}).get("DefaultVCpus", 0),
            spec.get("MemoryInfo", {}).get("SizeInMiB", 0),
        )

    return sorted(filter(_check_instance, instance_types), key=_sort_key)


def _resolve_git_package(package):
    match = REG_GITPACKAGE.match(package)
    if not match:
        raise ValueError(f"Invalid git package: {package}")
    repo_url, branch, subdirectory = match.groups()
    parsed = urlparse(repo_url)

    path_parts = [parsed.netloc] + parsed.path.split("/")

    return repo_url, branch, subdirectory, path_parts


def _get_it_card(spec):
    """
    InstanceType: g4dn.2xlarge
    VCpuInfo:
      DefaultCores: 32
      DefaultThreadsPerCore: 2
      DefaultVCpus: 64

    MemoryInfo:
      SizeInMiB: 32768

    GpuInfo:
      Gpus:
        - Count: 1
          Manufacturer: NVIDIA
          MemoryInfo:
            SizeInMiB: 16384
          Name: T4
      TotalGpuMemoryInMiB: 16384
    """
    return f"cpus: {spec['VCpuInfo']['DefaultVCpus']}, mem: {spec['MemoryInfo']['SizeInMiB']}, gpu: {spec['GpuInfo']['Gpus'][0]['Name']} x {spec['GpuInfo']['Gpus'][0]['Count']}, cost: $0.1/hour"


def _ensure_aws_security_group(group_name="cllama-http-default"):
    try:
        existing_groups = subprocess.check_output(
            [
                "aws",
                "ec2",
                "describe-security-groups",
                "--filters",
                f"Name=group-name,Values={group_name}",
                "--no-cli-pager",
            ]
        )
        existing_groups = json.loads(existing_groups)
        if existing_groups["SecurityGroups"]:
            return existing_groups["SecurityGroups"][0]["GroupId"]

        result = subprocess.check_output(
            [
                "aws",
                "ec2",
                "create-security-group",
                "--group-name",
                group_name,
                "--description",
                "Default VPC security group for cllama services",
                "--no-cli-pager",
            ]
        )
        result = json.loads(result)
        security_group_id = result["GroupId"]

        subprocess.check_call(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                security_group_id,
                "--protocol",
                "tcp",
                "--port",
                "80",
                "--cidr",
                "0.0.0.0/0",
                "--no-cli-pager",
            ]
        )
        subprocess.check_call(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                security_group_id,
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                "0.0.0.0/0",
                "--no-cli-pager",
            ]
        )
        subprocess.check_call(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                security_group_id,
                "--protocol",
                "tcp",
                "--port",
                "22",
                "--cidr",
                "0.0.0.0/0",
                "--no-cli-pager",
            ]
        )
        return security_group_id
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create security group: {e}")


@app.command()
def serve(model: str, tag: str = "latest", force_rebuild: bool = False):
    if ":" in model:
        model, tag = model.split(":")
    if tag == "latest":
        tag = next(iter(MODEL_INFOS[model].keys()))

    package = MODEL_INFOS[model][tag]
    repo, branch, subdirectory, path_parts = _resolve_git_package(package)
    repo_dir = REPO_DIR.joinpath(*path_parts)
    bento_project_dir = repo_dir / subdirectory

    if force_rebuild:
        shutil.rmtree(repo_dir, ignore_errors=True)

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            cmd = ["git", "clone", "--branch", branch, repo, str(repo_dir)]
            print(f"\n$ {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except:
            shutil.rmtree(repo_dir, ignore_errors=True)
            raise

    bento_info = get_bento_info(f"{model}:{tag}", bento_project_dir)

    if len(bento_info["services"]) != 1:
        raise ValueError("Only support one service currently")

    envs = {}
    if len(bento_info.get("envs", [])) > 0:
        for env in bento_info["envs"]:
            if env["name"] in os.environ:
                value = os.environ.get(env["name"])
                questionary.print(f"Using environment value for {env['name']}")
            elif env.get("value"):
                value = questionary.text(
                    f"Enter value for {env['name']}",
                    default=env["value"],
                ).ask()
            else:
                value = questionary.text(
                    f"Enter value for {env['name']}",
                ).ask()
            envs[env["name"]] = value

    cloud_provider = questionary.select(
        "Select a cloud provider",
        choices=[
            questionary.Choice(title="Local", value="aws"),
            questionary.Choice(title="BentoCloud", value="cloud"),
        ],
    ).ask()

    if cloud_provider == "cloud":
        cloud_provider = questionary.select(
            "You haven't logged in to BentoCloud, select an action",
            choices=[
                questionary.Choice(title="Login with Token", value="login"),
                questionary.Choice(title="Sign up ($10 free credit)", value="signup"),
            ],
        ).ask()
        if cloud_provider == "login":
            token = questionary.text("Enter your token").ask()
            cmd = ["bentoml", "cloud", "login", "--token", token]
            # print(f"\n$ {' '.join(cmd)}")
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to login")
        elif cloud_provider == "signup":
            token = questionary.text(
                "Open https://cloud.bentoml.org/signup in your browser",
            ).ask()
            # cmd = ["bentoml", "cloud", "signup"]
            # print(f"\n$ {' '.join(cmd)}")
            # try:
            # subprocess.check_call(cmd)
            # except subprocess.CalledProcessError:
            # raise RuntimeError("Failed to sign up")

    elif cloud_provider == "aws":
        try:
            cmd = ["aws", "ec2", "describe-instance-types", "--no-cli-pager"]
            print(f"\n$ {' '.join(cmd)}")
            _instance_types = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError:
            raise
            # print(e)
            # _cli_install_aws()
        available_it_infos = json.loads(_instance_types)["InstanceTypes"]
        # pyaml.p(available_it_infos)

        service = bento_info["services"][0]
        if "config" not in service or "resources" not in service["config"]:
            raise ValueError("Service config is missing")
        elif "gpu" in service["config"]["resources"]:
            gpu_count = service["config"]["resources"]["gpu"]
            gpu_type = service["config"]["resources"].get("gpu_type")
            gpu_memory = service["config"]["resources"].get("gpu_memory")
            supported_its = _filter_instance_types(
                available_it_infos,
                gpu_count,
                gpu_memory,
                gpu_type,
            )
            it = questionary.select(
                "Select an instance type",
                choices=[
                    questionary.Choice(
                        title=_get_it_card(it_info),
                        value=it_info["InstanceType"],
                    )
                    for it_info in supported_its
                ],
            ).ask()
            security_group_id = _ensure_aws_security_group()
            AMI = "ami-02623cf022763d4a1"

            init_script_file = TEMP_DIR / f"init_script_{str(uuid.uuid4())[:8]}.sh"
            with open(init_script_file, "w") as f:
                f.write(
                    INIT_SCRIPT_TEMPLATE.format(
                        repo=repo,
                        subdirectory=subdirectory,
                        model=model,
                        tag=tag,
                        env_args=" ".join([f"-e {k}={v}" for k, v in envs.items()]),
                    )
                )
            # grant permission
            os.chmod(init_script_file, 0o755)
            cmd = [
                "aws",
                "ec2",
                "run-instances",
                "--image-id",
                AMI,
                "--instance-type",
                it,
                "--security-group-ids",
                security_group_id,
                "--user-data",
                f"file://{init_script_file}",
                "--key-name",
                "jiang",
                "--count",
                "1",
                "--no-cli-pager",
            ]
            # print(f"\n$ {' '.join(cmd)}")
            try:
                result = subprocess.check_output(cmd)
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to create instance")
            result = json.loads(result)
            instance_id = result["Instances"][0]["InstanceId"]
            print(f"Deployment {instance_id} is created")

            cmd = [
                "aws",
                "ec2",
                "describe-instances",
                "--instance-ids",
                instance_id,
                "--no-cli-pager",
            ]
            # print(f"\n$ {' '.join(cmd)}")
            result = subprocess.check_output(cmd)
            result = json.loads(result)
            public_ip = result["Reservations"][0]["Instances"][0]["PublicIpAddress"]
            print(f"Public IP: {public_ip}")

            server_start_time = time.time()
            print("Server is starting...")
            with prompt_toolkit.shortcuts.ProgressBar() as pb:
                for _ in pb(range(100)):
                    start_time = time.time()
                    try:
                        with bentoml.SyncHTTPClient(f"http://{public_ip}"):
                            break
                    except Exception:
                        time.sleep(max(0, 6 - (time.time() - start_time)))
                else:
                    raise RuntimeError("Instance is not ready after 10 minutes")
            print(f"Server started in {time.time() - server_start_time:.2f} seconds")
            print(f"HTTP server is ready at http://{public_ip}")
            return
        else:
            raise ValueError("GPU is required for now")
    if cloud_provider == "bentocloud":
        cmd = ["bentoml", "cloud", "current-context"]
        # print(f"\n$ {' '.join(cmd)}")
        try:
            output = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "Failed to get bentocloud login context, please login first",
            )


@app.command()
def run(model: str, tag: str = "latest", force_rebuild: bool = False):
    serve(model, tag, force_rebuild)


INIT_SCRIPT_TEMPLATE = """#!/bin/bash
pip3 install bentoml
rm -r /usr/local/cuda*
git clone {repo} /root/bento_repo
export BENTOML_HOME=/root/bento_repo/{subdirectory}
bentoml containerize {model}:{tag} --image-tag {model}:{tag}
docker run --restart always --gpus all -d -p 80:3000 {env_args} {model}:{tag}

nvidia-smi -q | grep -A2 "ECC Mode" | grep "Current" | grep "Enabled"
ECC_ENABLED=$?

if [[ $ECC_ENABLED -eq 0 ]]; then
  echo "ECC is enabled. Disabling now..."
  nvidia-smi -e 0
  reboot
else
  echo "ECC is not enabled. No changes made."
fi
"""


if __name__ == "__main__":
    app()
