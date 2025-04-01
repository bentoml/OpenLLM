from __future__ import annotations

import json, os, pathlib, shutil, subprocess, typing
import typer

from openllm.analytic import OpenLLMTyper
from openllm.accelerator_spec import ACCELERATOR_SPECS
from openllm.common import INTERACTIVE, BentoInfo, DeploymentTarget, EnvVars, output, run_command

app = OpenLLMTyper()


def resolve_cloud_config() -> pathlib.Path:
    env = os.environ.get('BENTOML_HOME')
    if env is not None:
        return pathlib.Path(env) / '.yatai.yaml'
    return pathlib.Path.home() / 'bentoml' / '.yatai.yaml'


def _get_deploy_cmd(
    bento: BentoInfo, target: typing.Optional[DeploymentTarget] = None, cli_envs: typing.Optional[list[str]] = None
) -> tuple[list[str], EnvVars]:
    cmd = ['bentoml', 'deploy', bento.bentoml_tag]
    env = EnvVars({'BENTOML_HOME': f'{bento.repo.path}/bentoml'})

    # Process CLI env vars first to determine overrides
    explicit_envs: dict[str, str] = {}
    if cli_envs:
        for env_var in cli_envs:
            if '=' in env_var:
                name, value = env_var.split('=', 1)
                explicit_envs[name] = value
            else:
                name = env_var
                value = typing.cast(str, os.environ.get(name))
                if value is None:
                    output(f'Environment variable \'{name}\' specified via --env but not found in the current environment.', style='red')
                    raise typer.Exit(1)
                explicit_envs[name] = value

    # Process envs defined in bento.yaml, skipping those overridden by CLI
    required_envs = bento.bento_yaml.get('envs', [])
    required_env_names = [env['name'] for env in required_envs if 'name' in env and env['name'] not in explicit_envs]
    if required_env_names:
        output(
            f'This model requires the following environment variables to run (unless overridden via --env): {required_env_names!r}',
            style='yellow',
        )

    for env_info in required_envs:
        name = typing.cast(str, env_info.get('name'))
        if not name or name in explicit_envs:
            continue

        if os.environ.get(name):
            default = os.environ[name]
        elif 'value' in env_info:
            default = env_info['value']
        else:
            default = ''

        if INTERACTIVE.get():
            import questionary

            value = questionary.text(f'{name}: (from bento.yaml)', default=default).ask()
        else:
            if default == '':
                output(f'Environment variable {name} (from bento.yaml) is required but not provided', style='red')
                raise typer.Exit(1)
            else:
                value = default

        if value is None:
            raise typer.Exit(1)
        cmd += ['--env', f'{name}={value}']

    # Add explicitly provided env vars from CLI
    for name, value in explicit_envs.items():
        cmd += ['--env', f'{name}={value}']

    if target:
        cmd += ['--instance-type', target.name]

    base_config = resolve_cloud_config()
    if not base_config.exists():
        raise Exception('Cannot find cloud config.')
    # remove before copy
    if (bento.repo.path / 'bentoml' / '.yatai.yaml').exists():
        (bento.repo.path / 'bentoml' / '.yatai.yaml').unlink()
    shutil.copy(base_config, bento.repo.path / 'bentoml' / '.yatai.yaml')

    return cmd, env


def ensure_cloud_context() -> None:
    import questionary

    cmd = ['bentoml', 'cloud', 'current-context']
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        context = json.loads(result)
        output(f'  bentoml already logged in: {context["endpoint"]}', style='green', level=20)
    except subprocess.CalledProcessError:
        output('  bentoml not logged in', style='red')
        if not INTERACTIVE.get():
            output('\n  get bentoml logged in by:')
            output('    $ bentoml cloud login', style='orange')
            output('')
            output(
                """  * you may need to visit https://cloud.bentoml.com to get an account. you can also bring your own bentoml cluster (BYOC) to your team from https://bentoml.com/contact""",
                style='yellow',
            )
            raise typer.Exit(1)
        else:
            action = questionary.select(
                'Choose an action:', choices=['I have a BentoCloud account', 'get an account in two minutes']
            ).ask()
            if action is None:
                raise typer.Exit(1)
            elif action == 'get an account in two minutes':
                output('Please visit https://cloud.bentoml.com to get your token', style='yellow')
            endpoint = questionary.text('Enter the endpoint: (similar to https://my-org.cloud.bentoml.com)').ask()
            if endpoint is None:
                raise typer.Exit(1)
            token = questionary.text('Enter your token: (similar to cniluaxxxxxxxx)').ask()
            if token is None:
                raise typer.Exit(1)
            cmd = ['bentoml', 'cloud', 'login', '--api-token', token, '--endpoint', endpoint]
            try:
                result = subprocess.check_output(cmd)
                output('  Logged in successfully', style='green')
            except subprocess.CalledProcessError:
                output('  Failed to login', style='red')
                raise typer.Exit(1)


def get_cloud_machine_spec() -> list[DeploymentTarget]:
    ensure_cloud_context()
    cmd = ['bentoml', 'deployment', 'list-instance-types', '-o', 'json']
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        instance_types = json.loads(result)
        return [
            DeploymentTarget(
                source='cloud',
                name=it['name'],
                price=it['price'],
                platform='linux',
                accelerators=(
                    [ACCELERATOR_SPECS[it['gpu_type']] for _ in range(int(it['gpu']))]
                    if it.get('gpu') and it['gpu_type'] in ACCELERATOR_SPECS
                    else []
                ),
            )
            for it in instance_types
        ]
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        output('Failed to get cloud instance types', style='red')
        return []


def deploy(bento: BentoInfo, target: DeploymentTarget, cli_envs: typing.Optional[list[str]] = None) -> None:
    ensure_cloud_context()
    cmd, env = _get_deploy_cmd(bento, target, cli_envs=cli_envs)
    run_command(cmd, env=env, cwd=None)
