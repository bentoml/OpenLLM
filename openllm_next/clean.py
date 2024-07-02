import pathlib
import shutil

import questionary

from openllm_next.common import (
    REPO_DIR,
    VENV_DIR,
    VERBOSE_LEVEL,
    OpenLLMTyper,
    output,
)

app = OpenLLMTyper(help="clean up and release disk space used by OpenLLM")


HUGGINGFACE_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"


@app.command(help="Clean up all the cached models from huggingface")
def model_cache(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    used_space = sum(f.stat().st_size for f in HUGGINGFACE_CACHE.rglob("*"))
    sure = questionary.confirm(
        f"This will remove all models cached by Huggingface (~{used_space / 1024 / 1024:.2f}MB), are you sure?"
    ).ask()
    if not sure:
        return
    shutil.rmtree(HUGGINGFACE_CACHE, ignore_errors=True)
    output("All models cached by Huggingface have been removed", style="green")


@app.command(help="Clean up all the virtual environments created by OpenLLM")
def venvs(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    used_space = sum(f.stat().st_size for f in VENV_DIR.rglob("*"))
    sure = questionary.confirm(
        f"This will remove all virtual environments created by OpenLLM (~{used_space / 1024 / 1024:.2f}MB), are you sure?"
    ).ask()
    if not sure:
        return
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    output("All virtual environments have been removed", style="green")


@app.command(help="Clean up all the repositories cloned by OpenLLM")
def repos(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    output("All repositories have been removed", style="green")


@app.command(
    name="all",
    help="Clean up all above",
)
def all_cache(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    repos()
    venvs()
    model_cache()
