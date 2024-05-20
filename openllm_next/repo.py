import typer

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
)


app = typer.Typer()


GIT_REPO_RE = re.compile(
    r"git\+https://(?P<server>.+)/(?P<owner>.+)/(?P<repo>.+?)(@(?P<branch>.+))?$"
)


@app.command()
def list():
    config = load_config()
    pyaml.pprint(config.repos)


def parse_repo_url(repo_url):
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
    return (
        match.group("server"),
        match.group("owner"),
        match.group("repo"),
        match.group("branch") or "main",
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
    for name, repo in config.repos.items():
        server, owner, repo_name, branch = parse_repo_url(repo)
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
                run_command(cmd)
            except subprocess.CalledProcessError:
                shutil.rmtree(repo_dir, ignore_errors=True)
                questionary.print(f"Failed to clone repo {name}", style=ERROR_STYLE)
        else:
            try:
                cmd = ["git", "fetch", "origin", branch]
                run_command(cmd, cwd=repo_dir)
                cmd = ["git", "reset", "--hard", f"origin/{branch}"]
                run_command(cmd, cwd=repo_dir)
                cmd = ["git", "clean", "-fdx"]
                run_command(cmd, cwd=repo_dir)
            except:
                shutil.rmtree(repo_dir, ignore_errors=True)
                questionary.print(f"Failed to update repo {name}", style=ERROR_STYLE)
    for repo_dir in REPO_DIR.glob("*/*/*"):
        if tuple(repo_dir.parts[-3:]) not in repos_in_use:
            shutil.rmtree(repo_dir, ignore_errors=True)
            questionary.print(f"Removed unused repo {repo_dir}")
    questionary.print("Repos updated", style=SUCCESS_STYLE)


if __name__ == "__main__":
    app()
