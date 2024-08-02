import os
import pathlib
import shutil

import questionary

from openllm.analytic import OpenLLMTyper
from openllm.common import (CONFIG_FILE, REPO_DIR, VENV_DIR, VERBOSE_LEVEL,
                            output)

app = OpenLLMTyper(help='clean up and release disk space used by OpenLLM')

HUGGINGFACE_CACHE = pathlib.Path.home() / '.cache' / 'huggingface' / 'hub'


def _du(path: pathlib.Path) -> int:
    seen_paths = set()
    used_space = 0

    for f in path.rglob('*'):
        if os.name == 'nt':  # Windows system
            # On Windows, directly add file sizes without considering hard links
            used_space += f.stat().st_size
        else:
            # On non-Windows systems, use inodes to avoid double counting
            stat = f.stat()
            if stat.st_ino not in seen_paths:
                seen_paths.add(stat.st_ino)
                used_space += stat.st_size
    return used_space


@app.command(help='Clean up all the cached models from huggingface')
def model_cache(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    used_space = _du(HUGGINGFACE_CACHE)
    sure = questionary.confirm(
        f'This will remove all models cached by Huggingface (~{used_space / 1024 / 1024:.2f}MB), are you sure?'
    ).ask()
    if not sure:
        return
    shutil.rmtree(HUGGINGFACE_CACHE, ignore_errors=True)
    output('All models cached by Huggingface have been removed', style='green')


@app.command(help='Clean up all the virtual environments created by OpenLLM')
def venvs(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)

    used_space = _du(VENV_DIR)
    sure = questionary.confirm(
        f'This will remove all virtual environments created by OpenLLM (~{used_space / 1024 / 1024:.2f}MB), are you sure?'
    ).ask()
    if not sure:
        return
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    output('All virtual environments have been removed', style='green')


@app.command(help='Clean up all the repositories cloned by OpenLLM')
def repos(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    shutil.rmtree(REPO_DIR, ignore_errors=True)
    output('All repositories have been removed', style='green')


@app.command(help='Reset configurations to default')
def configs(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    shutil.rmtree(CONFIG_FILE, ignore_errors=True)
    output('All configurations have been reset', style='green')


@app.command(name='all', help='Clean up all above and bring OpenLLM to a fresh start')
def all_cache(verbose: bool = False):
    if verbose:
        VERBOSE_LEVEL.set(20)
    repos()
    venvs()
    model_cache()
    configs()
