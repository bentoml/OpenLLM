import datetime
import re
import shutil
import typing

from pathlib import Path
import os

import pyaml
import questionary
import typer

from openllm.analytic import OpenLLMTyper
from openllm.common import INTERACTIVE, REPO_DIR, VERBOSE_LEVEL, RepoInfo, load_config, output, save_config

UPDATE_INTERVAL = datetime.timedelta(days=3)
TEST_REPO = os.getenv('OPENLLM_TEST_REPO', None)  # for testing


app = OpenLLMTyper(help='manage repos')


@app.command(name='list', help='list available repo')
def cmd_list(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    pyaml.pprint(list_repo(), sort_dicts=False, sort_keys=False)


@app.command(name='remove', help='remove given repo')
def cmd_remove(name: str):
    if TEST_REPO:
        return
    config = load_config()
    if name not in config.repos:
        output(f'Repo {name} does not exist', style='red')
        return

    del config.repos[name]
    save_config(config)
    output(f'Repo {name} removed', style='green')


@app.command(name='update', help='update default repo')
def cmd_update():
    if TEST_REPO:
        return
    repos_in_use = set()
    for repo in list_repo():
        repos_in_use.add((repo.server, repo.owner, repo.repo, repo.branch))
        if repo.path.exists():
            shutil.rmtree(repo.path, ignore_errors=True)
        repo.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _clone_repo(repo)
            output('')
            output(f'Repo `{repo.name}` updated', style='green')
        except Exception as e:
            shutil.rmtree(repo.path, ignore_errors=True)
            output(f'Failed to clone repo {repo.name}', style='red')
            output(e)
    for c in REPO_DIR.glob('*/*/*/*'):
        repo_spec = tuple(c.parts[-4:])
        if repo_spec not in repos_in_use:
            shutil.rmtree(c, ignore_errors=True)
            output(f'Removed unused repo cache {c}')
    with open(REPO_DIR / 'last_update', 'w') as f:
        f.write(datetime.datetime.now().isoformat())
    for repo in list_repo():
        _complete_alias(repo.name)


@app.command(name='add', help='add new repo')
def cmd_add(name: str, repo: str):
    if TEST_REPO:
        return
    name = name.lower()
    if not name.isidentifier():
        output(f'Invalid repo name: {name}, should only contain letters, numbers and underscores', style='red')
        return

    try:
        parse_repo_url(repo)
    except ValueError:
        output(f'Invalid repo url: {repo}', style='red')
        return

    config = load_config()
    if name in config.repos:
        override = questionary.confirm(f'Repo {name} already exists({config.repos[name]}), override?').ask()
        if not override:
            return

    config.repos[name] = repo
    save_config(config)
    output(f'Repo {name} added', style='green')


@app.command(name='default', help='get default repo path')
def default():
    if TEST_REPO:
        return
    output((info := parse_repo_url(load_config().repos['default'], 'default')).path)
    return info.path


def list_repo(repo_name: typing.Optional[str] = None) -> typing.List[RepoInfo]:
    if TEST_REPO:
        return [
            RepoInfo(
                name='default', url='', server='test', owner='test', repo='test', branch='main', path=Path(TEST_REPO)
            )
        ]
    config = load_config()
    repos = []
    for _repo_name, repo_url in config.repos.items():
        if repo_name is not None and _repo_name != repo_name:
            continue
        repo = parse_repo_url(repo_url, _repo_name)
        repos.append(repo)
    return repos


def _complete_alias(repo_name: str):
    from openllm.model import list_bento

    for bento in list_bento(repo_name=repo_name):
        alias = bento.labels.get('aliases', '').strip()
        if alias:
            for a in alias.split(','):
                with open(bento.path.parent / a, 'w') as f:
                    f.write(bento.version)


def _clone_repo(repo: RepoInfo):
    """
    clone the repo to local, use `git` command if possible, otherwise use dulwich
    """
    import subprocess

    try:
        subprocess.run(['git', 'clone', '--depth=1', '-b', repo.branch, repo.url, str(repo.path)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        import dulwich
        import dulwich.porcelain

        dulwich.porcelain.clone(repo.url, str(repo.path), checkout=True, depth=1, branch=repo.branch)


def ensure_repo_updated():
    if TEST_REPO:
        return
    last_update_file = REPO_DIR / 'last_update'
    if not last_update_file.exists():
        if INTERACTIVE.get():
            choice = questionary.confirm(
                'The repo cache is never updated, do you want to update it to fetch the latest model list?'
            ).ask()
            if choice:
                cmd_update()
            return
        else:
            output(
                'The repo cache is never updated, please run `openllm repo update` to fetch the latest model list',
                style='red',
            )
            raise typer.Exit(1)
    last_update = datetime.datetime.fromisoformat(last_update_file.read_text().strip())
    if datetime.datetime.now() - last_update > UPDATE_INTERVAL:
        if INTERACTIVE.get():
            choice = questionary.confirm(
                'The repo cache is outdated, do you want to update it to fetch the latest model list?'
            ).ask()
            if choice:
                cmd_update()
        else:
            output(
                'The repo cache is outdated, please run `openllm repo update` to fetch the latest model list',
                style='yellow',
            )


GIT_HTTP_RE = re.compile(
    r'(?P<schema>git|ssh|http|https):\/\/(?P<server>[\.\w\d\-]+)\/(?P<owner>[\w\d\-]+)\/(?P<repo>[\w\d\-\_\.]+)(@(?P<branch>.+))?(\/)?$'
)
GIT_SSH_RE = re.compile(
    r'git@(?P<server>[\.\w\d-]+):(?P<owner>[\w\d\-]+)\/(?P<repo>[\w\d\-\_\.]+)(@(?P<branch>.+))?(\/)?$'
)


def parse_repo_url(repo_url: str, repo_name: typing.Optional[str] = None) -> RepoInfo:
    """
    parse the git repo url to server, owner, repo name, branch
    >>> parse_repo_url('https://github.com/bentoml/bentovllm@main')
    ('github.com', 'bentoml', 'bentovllm', 'main')

    >>> parse_repo_url('https://github.com/bentoml/bentovllm.git@main')
    ('github.com', 'bentoml', 'bentovllm', 'main')

    >>> parse_repo_url('https://github.com/bentoml/bentovllm')
    ('github.com', 'bentoml', 'bentovllm', 'main')

    >>> parse_repo_url('git@github.com:bentoml/openllm-models.git')
    ('github.com', 'bentoml', 'openllm-models', 'main')
    """
    match = GIT_HTTP_RE.match(repo_url)
    if match:
        schema = match.group('schema')
    else:
        match = GIT_SSH_RE.match(repo_url)
        if not match:
            raise ValueError(f'Invalid git repo url: {repo_url}')
        schema = None

    if match.group('branch') is not None:
        repo_url = repo_url[: match.start('branch') - 1]

    server = match.group('server')
    owner = match.group('owner')
    repo = match.group('repo')
    if repo.endswith('.git'):
        repo = repo[:-4]
    branch = match.group('branch') or 'main'

    if schema is not None:
        repo_url = f'{schema}://{server}/{owner}/{repo}'
    else:
        repo_url = f'git@{server}:{owner}/{repo}'

    path = REPO_DIR / server / owner / repo / branch
    return RepoInfo(
        name=repo if repo_name is None else repo_name,
        url=repo_url,
        server=server,
        owner=owner,
        repo=repo,
        branch=branch,
        path=path,
    )


if __name__ == '__main__':
    app()
