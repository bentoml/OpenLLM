import typer
import pathlib

import shutil
import questionary
import re
import subprocess
import pyaml
from openllm_next.common import (
    ERROR_STYLE,
    SUCCESS_STYLE,
    REPO_DIR,
    load_config,
    save_config,
    run_command,
    RepoInfo,
)


app = typer.Typer()


GIT_REPO_RE = re.compile(
    r"git\+https://(?P<server>.+)/(?P<owner>.+)/(?P<repo>.+?)(@(?P<branch>.+))?$"
)


@app.command()
def list():
    config = load_config()
    pyaml.pprint(config.repos)


def parse_repo_url(repo_url, repo_name=None) -> RepoInfo:
    """
    parse the git repo url to server, owner, repo name, branch
    >>> parse_repo_url("git+https://github.com/bojiang/bentovllm@main")
    ('github.com', 'bojiang', 'bentovllm', 'main')

    >>> parse_repo_url("git+https://github.com/bojiang/bentovllm")
    ('github.com', 'bojiang', 'bentovllm', 'main')
    """
    match = GIT_REPO_RE.match(repo_url)
    if not match:
        raise ValueError(f"Invalid git repo url: {repo_url}")
    server = match.group("server")
    owner = match.group("owner")
    repo = match.group("repo")
    branch = match.group("branch") or "main"
    cache_path = REPO_DIR / server / owner / repo
    return RepoInfo(
        name=repo if repo_name is None else repo_name,
        url=repo_url,
        server=server,
        owner=owner,
        repo=repo,
        branch=branch,
        cache_path=cache_path,
    )


@app.command()
def add(name: str, repo: str):
    name = name.lower()
    if not name.isidentifier():
        questionary.print(
            f"Invalid repo name: {name}, should only contain letters, numbers and underscores",
            style=ERROR_STYLE,
        )
        return

    config = load_config()
    if name in config.repos:
        override = questionary.confirm(
            f"Repo {name} already exists({config.repos[name]}), override?"
        ).ask()
        if not override:
            return

    config.repos[name] = repo
    save_config(config)
    pyaml.pprint(config.repos)


@app.command()
def remove(name: str):
    config = load_config()
    if name not in config.repos:
        questionary.print(f"Repo {name} does not exist", style=ERROR_STYLE)
        return

    del config.repos[name]
    save_config(config)
    pyaml.pprint(config.repos)


@app.command()
def update():
    config = load_config()
    repos_in_use = set()
    for repo_name, repo in config.repos.items():
        repo = parse_repo_url(repo, repo_name)
        repos_in_use.add((repo.server, repo.owner, repo.repo))
        if not repo.cache_path.exists():
            repo.cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cmd = [
                    "git",
                    "clone",
                    "--branch",
                    repo.branch,
                    f"https://{repo.server}/{repo.owner}/{repo.repo}.git",
                    str(repo.cache_path),
                ]
                run_command(cmd)
            except subprocess.CalledProcessError:
                shutil.rmtree(repo.cache_path, ignore_errors=True)
                questionary.print(f"Failed to clone repo {repo.name}", style=ERROR_STYLE)
        else:
            try:
                cmd = ["git", "fetch", "origin", repo.branch]
                run_command(cmd, cwd=repo.cache_path)
                cmd = ["git", "reset", "--hard", f"origin/{repo.branch}"]
                run_command(cmd, cwd=repo.cache_path)
                cmd = ["git", "clean", "-fdx"]
                run_command(cmd, cwd=repo.cache_path)
            except:
                shutil.rmtree(repo.cache_path, ignore_errors=True)
                questionary.print(f"Failed to update repo {repo.name}", style=ERROR_STYLE)
    for c in REPO_DIR.glob("*/*/*"):
        if tuple(c.parts[-3:]) not in repos_in_use:
            shutil.rmtree(c, ignore_errors=True)
            questionary.print(f"Removed unused repo cache {c}")
    questionary.print("Repos updated", style=SUCCESS_STYLE)


if __name__ == "__main__":
    app()
