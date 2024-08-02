import functools
import os
import pathlib
import shutil

import typer

from openllm.common import (
    VENV_DIR,
    VERBOSE_LEVEL,
    BentoInfo,
    VenvSpec,
    output,
    run_command,
)


@functools.lru_cache
def _resolve_bento_env_spec(bento: BentoInfo):
    ver_file = bento.path / "env" / "python" / "version.txt"
    assert ver_file.exists(), f"cannot find version file in {bento.path}"

    lock_file = bento.path / "env" / "python" / "requirements.lock.txt"
    if not lock_file.exists():
        lock_file = bento.path / "env" / "python" / "requirements.txt"

    ver = ver_file.read_text().strip()
    reqs = lock_file.read_text().strip()

    return VenvSpec(
        python_version=ver,
        requirements_txt=reqs,
        name_prefix=f"{bento.tag.replace(':', '_')}-1-",
    )


def _ensure_venv(
    env_spec: VenvSpec,
) -> pathlib.Path:
    venv = VENV_DIR / str(hash(env_spec))
    if venv.exists() and not (venv / "DONE").exists():
        shutil.rmtree(venv, ignore_errors=True)
    if not venv.exists():
        output(f"Installing model dependencies({venv})...", style="green")

        venv_py = (
            venv / "Scripts" / "python.exe"
            if os.name == "nt"
            else venv / "bin" / "python"
        )
        try:
            run_command(
                ["python", "-m", "uv", "venv", venv], silent=VERBOSE_LEVEL.get() < 10
            )
            with open(venv / "requirements.txt", "w") as f:
                f.write(env_spec.normalized_requirements_txt)
            run_command(
                [
                    "python",
                    "-m",
                    "uv",
                    "pip",
                    "install",
                    "-p",
                    str(venv_py),
                    "-r",
                    venv / "requirements.txt",
                ],
                silent=VERBOSE_LEVEL.get() < 10,
            )
            with open(venv / "DONE", "w") as f:
                f.write("DONE")
        except Exception as e:
            shutil.rmtree(venv, ignore_errors=True)
            if VERBOSE_LEVEL.get() >= 10:
                output(e, style="red")
            output(
                f"Failed to install dependencies to {venv}. Cleaned up.", style="red"
            )
            raise typer.Exit(1)
        output(f"Successfully installed dependencies to {venv}.", style="green")
        return venv
    else:
        return venv


def ensure_venv(bento: BentoInfo) -> pathlib.Path:
    env_spec = _resolve_bento_env_spec(bento)
    venv = _ensure_venv(env_spec)
    assert venv is not None
    return venv


def check_venv(bento: BentoInfo) -> bool:
    env_spec = _resolve_bento_env_spec(bento)
    venv = VENV_DIR / str(hash(env_spec))
    if not venv.exists():
        return False
    if venv.exists() and not (venv / "DONE").exists():
        return False
    return True
