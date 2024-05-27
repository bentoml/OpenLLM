import functools
import pathlib
import shutil
from types import SimpleNamespace
from typing import Iterable

import questionary
import typer

from openllm_next.common import VENV_DIR, VERBOSE_LEVEL, BentoInfo, md5, run_command


def _resolve_packages(requirement: str | pathlib.Path) -> dict[str, str]:
    from pip_requirements_parser import RequirementsFile

    requirements_txt = RequirementsFile.from_file(
        str(requirement),
        include_nested=True,
    )
    deps: dict[str, str] = {}
    for req in requirements_txt.requirements:
        if (
            req.is_editable
            or req.is_local_path
            or req.is_url
            or req.is_wheel
            or not req.name
            or not req.specifier
        ):
            continue
        for sp in req.specifier:
            if sp.operator == "==":
                assert req.line is not None
                deps[req.name] = req.line
                break
    return deps


class EnvSpec(SimpleNamespace):
    python_version: str
    python_packages: dict[str, str]
    name_prefix = ""

    def __hash__(self):
        return md5(
            # self.python_version,
            *sorted(self.python_packages.values()),
        )


@functools.lru_cache
def _resolve_bento_env_specs(bento: BentoInfo):
    ver_file = bento.path / "env" / "python" / "version.txt"
    assert ver_file.exists(), f"cannot find version file in {bento.path}"

    lock_file = bento.path / "env" / "python" / "requirements.lock.txt"
    if not lock_file.exists():
        lock_file = bento.path / "env" / "python" / "requirements.txt"

    python_packages = _resolve_packages(lock_file)
    PREHEAT_PIP_PACKAGES = ["torch", "vllm"]
    preheat_packages = {
        k: v for k, v in python_packages.items() if k in PREHEAT_PIP_PACKAGES
    }
    ver = ver_file.read_text().strip()
    return (
        EnvSpec(
            python_version=ver,
            python_packages=preheat_packages,
            name_prefix=f"{bento.tag.replace(':', '_')}-1-",
        ),
        EnvSpec(
            python_version=ver,
            python_packages=python_packages,
            name_prefix=f"{bento.tag.replace(':', '_')}-2-",
        ),
    )


def _ensure_venv(
    env_spec: EnvSpec, parrent_venv: pathlib.Path | None = None
) -> pathlib.Path:
    venv = VENV_DIR / str(hash(env_spec))
    if not venv.exists():
        questionary.print(f"Installing model dependencies({venv})...", style="green")
        try:
            run_command(["python", "-m", "venv", venv], silent=VERBOSE_LEVEL.get() < 1)
            pyver = next(venv.glob("lib/python*")).name
            if parrent_venv is not None:
                with open(
                    venv / "lib" / pyver / "site-packages" / f"{parrent_venv.name}.pth",
                    "w+",
                ) as f:
                    f.write(str(parrent_venv / "lib" / pyver / "site-packages"))
            with open(venv / "requirements.txt", "w") as f:
                f.write("\n".join(sorted(env_spec.python_packages.values())))
            run_command(
                [
                    venv / "bin" / "pip",
                    "install",
                    "-r",
                    venv / "requirements.txt",
                    "--upgrade-strategy",
                    "only-if-needed",
                ],
                silent=VERBOSE_LEVEL.get() < 1,
            )
            run_command(
                [
                    venv / "bin" / "pip",
                    "install",
                    "bentoml",
                    "--upgrade-strategy",
                    "only-if-needed",
                    "--upgrade",
                ],
                silent=VERBOSE_LEVEL.get() < 1,
            )
        except Exception:
            shutil.rmtree(venv, ignore_errors=True)
            questionary.print(
                f"Failed to install dependencies to {venv}. Cleaned up.",
                style="red",
            )
            raise typer.Exit(1)
        questionary.print(
            f"Successfully installed dependencies to {venv}.", style="green"
        )
        return venv
    else:
        return venv


def _ensure_venvs(env_spec_list: Iterable[EnvSpec]) -> pathlib.Path:
    last_venv = None
    for env_spec in env_spec_list:
        last_venv = _ensure_venv(env_spec, last_venv)
    assert last_venv is not None
    return last_venv


def ensure_venv(bento: BentoInfo) -> pathlib.Path:
    return _ensure_venvs(_resolve_bento_env_specs(bento))
