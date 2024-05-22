import re
import shutil
import subprocess

import pyaml
import questionary
import typer

from openllm_next.common import (
    ERROR_STYLE,
    REPO_DIR,
    SUCCESS_STYLE,
    RepoInfo,
    load_config,
    run_command,
    save_config,
)

app = typer.Typer()


@app.command()
def list():
    config = load_config()
    pyaml.pprint(
        [parse_repo_url(repo, name) for name, repo in config.repos.items()],
        sort_dicts=False,
        sort_keys=False,
    )


@app.command()
def remove(name: str):
    config = load_config()
    if name not in config.repos:
        questionary.print(f"Repo {name} does not exist", style=ERROR_STYLE)
        return

    del config.repos[name]
    save_config(config)
    questionary.print(f"Repo {name} removed", style=SUCCESS_STYLE)


@app.command()
def update():
    config = load_config()
    repos_in_use = set()
    for repo_name, repo in config.repos.items():
        repo = parse_repo_url(repo, repo_name)
        repos_in_use.add((repo.server, repo.owner, repo.repo))
        if not repo.path.exists():
            repo.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cmd = [
                    "git",
                    "clone",
                    "--branch",
                    repo.branch,
                    f"https://{repo.server}/{repo.owner}/{repo.repo}.git",
                    str(repo.path),
                ]
                run_command(cmd)
            except subprocess.CalledProcessError:
                shutil.rmtree(repo.path, ignore_errors=True)
                questionary.print(
                    f"Failed to clone repo {repo.name}", style=ERROR_STYLE
                )
        else:
            try:
                cmd = ["git", "fetch", "origin", repo.branch]
                run_command(cmd, cwd=repo.path)
                cmd = ["git", "reset", "--hard", f"origin/{repo.branch}"]
                run_command(cmd, cwd=repo.path)
                cmd = ["git", "clean", "-fdx"]
                run_command(cmd, cwd=repo.path)
            except:
                shutil.rmtree(repo.path, ignore_errors=True)
                questionary.print(
                    f"Failed to update repo {repo.name}", style=ERROR_STYLE
                )
    for c in REPO_DIR.glob("*/*/*"):
        if tuple(c.parts[-3:]) not in repos_in_use:
            shutil.rmtree(c, ignore_errors=True)
            questionary.print(f"Removed unused repo cache {c}")
    questionary.print("Repos updated", style=SUCCESS_STYLE)


GIT_REPO_RE = re.compile(
    r"git\+https://(?P<server>.+)/(?P<owner>.+)/(?P<repo>.+?)(@(?P<branch>.+))?$"
)


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
    path = REPO_DIR / server / owner / repo
    return RepoInfo(
        name=repo if repo_name is None else repo_name,
        url=repo_url,
        server=server,
        owner=owner,
        repo=repo,
        branch=branch,
        path=path,
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
    questionary.print(f"Repo {name} added", style=SUCCESS_STYLE)


if __name__ == "__main__":
    app()
