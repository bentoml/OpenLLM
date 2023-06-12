# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Any build-related utilities. This is used for CI.
"""
from __future__ import annotations

import importlib.metadata
import logging
import os
import typing as t
from pathlib import Path

import bentoml
import fs
import inflection
from bentoml._internal.bento.build_config import DockerOptions, PythonOptions
from bentoml._internal.configuration import get_debug_mode

import openllm
import openllm.utils as utils
from openllm.utils import pkg

if t.TYPE_CHECKING:
    from fs.base import FS

logger = logging.getLogger(__name__)

OPENLLM_DEV_BUILD = "OPENLLM_DEV_BUILD"


def build_editable(path: str) -> str | None:
    """Build OpenLLM if the OPENLLM_DEV_BUILD environment variable is set."""
    if str(os.environ.get(OPENLLM_DEV_BUILD, False)).lower() != "true":
        return

    # We need to build the package in editable mode, so that we can import it
    from build import ProjectBuilder
    from build.env import IsolatedEnvBuilder

    module_location = pkg.source_locations("openllm")
    if not module_location:
        raise RuntimeError(
            "Could not find the source location of OpenLLM. Make sure to unset"
            " OPENLLM_DEV_BUILD if you are developing OpenLLM."
        )
    pyproject_path = Path(module_location).parent.parent / "pyproject.toml"
    if os.path.isfile(pyproject_path.__fspath__()):
        logger.info("OpenLLM is installed in editable mode. Generating built wheels...")
        with IsolatedEnvBuilder() as env:
            builder = ProjectBuilder(pyproject_path.parent)
            builder.python_executable = env.executable
            builder.scripts_dir = env.scripts_dir
            env.install(builder.build_system_requires)
            return builder.build("wheel", path, config_settings={"--global-option": "--quiet"})
    raise RuntimeError(
        "Custom OpenLLM build is currently not supported. Please install OpenLLM from PyPI or built it from Git source."
    )


def construct_python_options(llm: openllm.LLM[t.Any, t.Any], llm_fs: FS) -> PythonOptions:
    # NOTE: add openllm to the default dependencies
    # if users has openllm custom built wheels, it will still respect
    # that since bentoml will always install dependencies from requirements.txt
    # first, then proceed to install everything inside the wheels/ folder.
    packages: list[str] = ["openllm"]

    if llm.config.__openllm_requirements__ is not None:
        packages.extend(llm.config.__openllm_requirements__)

    if not (str(os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD", False)).lower() == "false"):
        packages.append(f"bentoml>={'.'.join([str(i) for i in pkg.pkg_version_info('bentoml')])}")

    to_use_framework = llm.config.__openllm_env__.get_framework_env()
    if to_use_framework == "flax":
        assert (
            utils.is_flax_available()
        ), f"Flax is not available, while {llm.config.__openllm_env__.framework} is set to 'flax'"
        packages.extend(
            [
                f"flax>={importlib.metadata.version('flax')}",
                f"jax>={importlib.metadata.version('jax')}",
                f"jaxlib>={importlib.metadata.version('jaxlib')}",
            ]
        )
    elif to_use_framework == "tf":
        assert (
            utils.is_tf_available()
        ), f"TensorFlow is not available, while {llm.config.__openllm_env__.framework} is set to 'tf'"
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for candidate in candidates:
            try:
                _tf_version = importlib.metadata.version(candidate)
                packages.extend([f"tensorflow>={_tf_version}"])
                break
            except importlib.metadata.PackageNotFoundError:
                pass
    else:
        assert utils.is_torch_available(), "PyTorch is not available. Make sure to have it locally installed."
        packages.extend([f"torch>={importlib.metadata.version('torch')}"])

    wheels: list[str] = []
    built_wheels = build_editable(llm_fs.getsyspath("/"))
    if built_wheels is not None:
        wheels.append(llm_fs.getsyspath(f"/{built_wheels.split('/')[-1]}"))

    return PythonOptions(packages=packages, wheels=wheels, lock_packages=True)


def construct_docker_options(llm: openllm.LLM[t.Any, t.Any], _: FS, workers_per_resource: int | float) -> DockerOptions:
    _bentoml_config_options = os.environ.pop("BENTOML_CONFIG_OPTIONS", "")
    _bentoml_config_options_opts = [
        "api_server.traffic.timeout=3600",  # NOTE: Currently we hardcode this value
        f'runners."llm-{llm.config.__openllm_start_name__}-runner".traffic.timeout={llm.config.__openllm_timeout__}',
        f'runners."llm-{llm.config.__openllm_start_name__}-runner".workers_per_resource={workers_per_resource}',
    ]
    _bentoml_config_options += " " if _bentoml_config_options else "" + " ".join(_bentoml_config_options_opts)
    return DockerOptions(
        cuda_version="11.6",  # NOTE: Torch 2.0 currently only support 11.6 as the latest CUDA version
        env={
            llm.config.__openllm_env__.framework: llm.config.__openllm_env__.get_framework_env(),
            "OPENLLM_MODEL": llm.config.__openllm_model_name__,
            "OPENLLM_MODEL_ID": llm._model_id,
            "BENTOML_DEBUG": str(get_debug_mode()),
            "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
        },
        system_packages=["git"],
    )


@t.overload
def build(model_name: str, *, __cli__: t.Literal[False] = ..., **attrs: t.Any) -> bentoml.Bento:
    ...


@t.overload
def build(model_name: str, *, __cli__: t.Literal[True] = ..., **attrs: t.Any) -> tuple[bentoml.Bento, bool]:
    ...


def build(model_name: str, *, __cli__: bool = False, **attrs: t.Any) -> tuple[bentoml.Bento, bool] | bentoml.Bento:
    """Package a LLM into a Bento."""

    overwrite_existing_bento = attrs.pop("_overwrite_existing_bento", False)
    current_model_envvar = os.environ.pop("OPENLLM_MODEL", None)
    current_model_id_envvar = os.environ.pop("OPENLLM_MODEL_ID", None)
    _previously_built = False
    workers = attrs.pop("_workers", None)
    model_id: str = attrs.pop("model_id", None)

    llm_config = openllm.AutoConfig.for_model(model_name)

    logger.info("Packing '%s' into a Bento with kwargs=%s...", model_name, attrs)

    # NOTE: We set this environment variable so that our service.py logic won't raise RuntimeError
    # during build. This is a current limitation of bentoml build where we actually import the service.py into sys.path
    try:
        os.environ["OPENLLM_MODEL"] = inflection.underscore(model_name)
        os.environ["OPENLLM_MODEL_ID"] = model_id

        to_use_framework = llm_config.__openllm_env__.get_framework_env()
        if to_use_framework == "flax":
            llm = openllm.AutoFlaxLLM.for_model(model_name, model_id=model_id, llm_config=llm_config, **attrs)
        elif to_use_framework == "tf":
            llm = openllm.AutoTFLLM.for_model(model_name, model_id=model_id, llm_config=llm_config, **attrs)
        else:
            llm = openllm.AutoLLM.for_model(model_name, model_id=model_id, llm_config=llm_config, **attrs)

        labels = dict(llm.identifying_params)
        labels.update({"_type": llm.llm_type, "_framework": to_use_framework})
        service_name = f"generated_{llm.config.__openllm_model_name__}_service.py"
        workers_per_resource = utils.first_not_none(workers, default=llm.config.__openllm_workers_per_resource__)

        with fs.open_fs(f"temp://llm_{llm.config.__openllm_model_name__}") as llm_fs:
            # add service.py definition to this temporary folder
            utils.codegen.write_service(model_name, service_name, llm_fs)

            bento_tag = bentoml.Tag.from_taglike(f"{model_name}-service:{llm.tag.version}")
            try:
                bento = bentoml.get(bento_tag)
                if overwrite_existing_bento:
                    bentoml.delete(bento_tag)
                    raise bentoml.exceptions.NotFound("Overwriting previously saved Bento.")
                _previously_built = True
            except bentoml.exceptions.NotFound:
                logger.info("Building Bento for LLM '%s'", llm.config.__openllm_start_name__)
                bento = bentoml.bentos.build(
                    f"{service_name}:svc",
                    name=bento_tag.name,
                    labels=labels,
                    description=f"OpenLLM service for {llm.config.__openllm_start_name__}",
                    include=[
                        f for f in llm_fs.walk.files(filter=["*.py"])
                    ],  # NOTE: By default, we are using _service.py as the default service, for now.
                    exclude=["/venv", "__pycache__/", "*.py[cod]", "*$py.class"],
                    python=construct_python_options(llm, llm_fs),
                    docker=construct_docker_options(llm, llm_fs, workers_per_resource),
                    version=bento_tag.version,
                    build_ctx=llm_fs.getsyspath("/"),
                )
            if __cli__:
                return bento, _previously_built
            else:
                return bento
    except Exception as e:
        logger.error("\nException caught during building LLM %s: \n", model_name, exc_info=e)
        raise
    finally:
        del os.environ["OPENLLM_MODEL"]
        del os.environ["OPENLLM_MODEL_ID"]
        # restore original OPENLLM_MODEL envvar if set.
        if current_model_envvar is not None:
            os.environ["OPENLLM_MODEL"] = current_model_envvar
        if current_model_id_envvar is not None:
            os.environ["OPENLLM_MODEL_ID"] = current_model_id_envvar
