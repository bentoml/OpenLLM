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
import re
import subprocess
import sys
import typing as t
from pathlib import Path

import fs
import fs.copy
import orjson
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelStore

from .utils import DEBUG
from .utils import EnvVarMixin
from .utils import codegen
from .utils import is_flax_available
from .utils import is_tf_available
from .utils import is_torch_available
from .utils import pkg
from .utils import resolve_user_filepath


if t.TYPE_CHECKING:
    from fs.base import FS

    from bentoml._internal.bento import BentoStore

    from .models.auto.factory import _BaseAutoLLMClass

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


def construct_python_options(
    llm: openllm.LLM[t.Any, t.Any],
    llm_fs: FS,
    extra_dependencies: tuple[str, ...] | None = None,
    adapter_map: dict[str, str | None] | None = None,
) -> PythonOptions:
    packages = ["openllm"]
    if adapter_map is not None:
        packages += ["openllm[fine-tune]"]
    # NOTE: add openllm to the default dependencies
    # if users has openllm custom built wheels, it will still respect
    # that since bentoml will always install dependencies from requirements.txt
    # first, then proceed to install everything inside the wheels/ folder.
    if extra_dependencies is not None:
        filtered = set(extra_dependencies + ("fine-tune",))
        packages += [f"openllm[{k}]" for k in filtered]

    req = llm.config["requirements"]
    if req is not None:
        packages.extend(req)

    if str(os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD", False)).lower() == "false":
        packages.append(f"bentoml>={'.'.join([str(i) for i in pkg.pkg_version_info('bentoml')])}")

    env: EnvVarMixin = llm.config["env"]
    framework_envvar = env["framework_value"]
    if framework_envvar == "flax":
        assert is_flax_available(), f"Flax is not available, while {env.framework} is set to 'flax'"
        packages.extend(
            [
                f"flax>={importlib.metadata.version('flax')}",
                f"jax>={importlib.metadata.version('jax')}",
                f"jaxlib>={importlib.metadata.version('jaxlib')}",
            ]
        )
    elif framework_envvar == "tf":
        assert is_tf_available(), f"TensorFlow is not available, while {env.framework} is set to 'tf'"
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
        assert is_torch_available(), "PyTorch is not available. Make sure to have it locally installed."
        packages.extend([f"torch>={importlib.metadata.version('torch')}"])

    wheels: list[str] = []
    built_wheels = build_editable(llm_fs.getsyspath("/"))
    if built_wheels is not None:
        wheels.append(llm_fs.getsyspath(f"/{built_wheels.split('/')[-1]}"))

    return PythonOptions(packages=packages, wheels=wheels, lock_packages=True)


def construct_docker_options(
    llm: openllm.LLM[t.Any, t.Any],
    _: FS,
    workers_per_resource: int | float,
    quantize: t.LiteralString | None,
    bettertransformer: bool | None,
    adapter_map: dict[str, str | None] | None,
) -> DockerOptions:
    _bentoml_config_options = os.environ.pop("BENTOML_CONFIG_OPTIONS", "")
    _bentoml_config_options_opts = [
        "api_server.traffic.timeout=36000",  # NOTE: Currently we hardcode this value
        f'runners."llm-{llm.config["start_name"]}-runner".traffic.timeout={llm.config["timeout"]}',
        f'runners."llm-{llm.config["start_name"]}-runner".workers_per_resource={workers_per_resource}',
    ]
    _bentoml_config_options += " " if _bentoml_config_options else "" + " ".join(_bentoml_config_options_opts)
    env: EnvVarMixin = llm.config["env"]

    env_dict = {
        env.framework: env.framework_value,
        env.config: f"'{llm.config.model_dump_json().decode()}'",
        "OPENLLM_MODEL": llm.config["model_name"],
        "OPENLLM_MODEL_ID": llm.model_id,
        "OPENLLM_ADAPTER_MAP": f"'{orjson.dumps(adapter_map).decode()}'",
        "BENTOML_DEBUG": str(get_debug_mode()),
        "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
    }

    if adapter_map:
        env_dict["BITSANDBYTES_NOWELCOME"] = os.environ.get("BITSANDBYTES_NOWELCOME", "1")

    # We need to handle None separately here, as env from subprocess doesn't
    # accept None value.
    _env = EnvVarMixin(llm.config["model_name"], bettertransformer=bettertransformer, quantize=quantize)

    if _env.bettertransformer_value is not None:
        env_dict[_env.bettertransformer] = _env.bettertransformer_value
    if _env.quantize_value is not None:
        env_dict[_env.quantize] = _env.quantize_value

    # NOTE: Torch 2.0 currently only support 11.6 as the latest CUDA version
    return DockerOptions(cuda_version="11.6", env=env_dict, system_packages=["git"])


def create_bento(
    bento_tag: bentoml.Tag,
    llm_fs: FS,
    llm: openllm.LLM[t.Any, t.Any],
    workers_per_resource: int | float,
    quantize: t.LiteralString | None,
    bettertransformer: bool | None,
    adapter_map: dict[str, str | None] | None = None,
    extra_dependencies: tuple[str, ...] | None = None,
    build_ctx: str | None = None,
) -> bentoml.Bento:
    framework_envvar = llm.config["env"]["framework_value"]
    labels = dict(llm.identifying_params)
    labels.update({"_type": llm.llm_type, "_framework": framework_envvar, "start_name": llm.config["start_name"]})

    if adapter_map:
        labels.update(adapter_map)

    logger.info("Building Bento for '%s'", llm.config["start_name"])

    if adapter_map is not None:
        assert build_ctx is not None, "build_ctx is required when 'adapter_map' is not None"
        updated_mapping: dict[str, str | None] = {}
        for adapter_id, name in adapter_map.items():
            try:
                resolve_user_filepath(adapter_id, build_ctx)
                src_folder_name = os.path.basename(adapter_id)
                src_fs = fs.open_fs(build_ctx)
                llm_fs.makedir(src_folder_name, recreate=True)
                fs.copy.copy_dir(src_fs, adapter_id, llm_fs, src_folder_name)
                updated_mapping[src_folder_name] = name
            except FileNotFoundError:
                # this is the remote adapter, then just added back
                # note that there is a drawback here. If the path of the local adapter
                # path have the same name as the remote, then we currently don't support
                # that edge case.
                updated_mapping[adapter_id] = name
        adapter_map = updated_mapping

    # add service.py definition to this temporary folder
    codegen.write_service(llm, adapter_map, llm_fs)

    build_config = BentoBuildConfig(
        service=f"{llm.config['service_name']}:svc",
        name=bento_tag.name,
        labels=labels,
        description=f"OpenLLM service for {llm.config['start_name']}",
        include=list(llm_fs.walk.files()),
        exclude=["/venv", "/.venv", "__pycache__/", "*.py[cod]", "*$py.class"],
        python=construct_python_options(llm, llm_fs, extra_dependencies, adapter_map),
        docker=construct_docker_options(llm, llm_fs, workers_per_resource, quantize, bettertransformer, adapter_map),
    )

    bento = bentoml.Bento.create(
        build_config=build_config,
        version=bento_tag.version,
        build_ctx=llm_fs.getsyspath("/"),
    )

    # Now we have to format the model_id accordingly based on the model_fs
    model_type = bento.info.labels["_type"]
    model_framework = bento.info.labels["_framework"]
    model_store = ModelStore(bento._fs.opendir("models"))
    # the models should have the type
    try:
        model = model_store.get(f"{model_framework}-{model_type}")
    except bentoml.exceptions.NotFound:
        raise openllm.exceptions.OpenLLMException(f"Failed to find models for {llm.config['start_name']}")

    model_id_path = bento._fs.getsyspath(fs.path.join("models", model.tag.path()))
    service_fs_path = fs.path.join("src", llm.config["service_name"])
    service_path = bento._fs.getsyspath(service_fs_path)
    with open(service_path, "r") as f:
        service_contents = f.readlines()

    for it in service_contents:
        if codegen.OPENLLM_MODEL_ID in it:
            service_contents[service_contents.index(it)] = (
                codegen.ModelIdFormatter(fs.path.relativefrom(bento._fs.getsyspath("/src"), model_id_path)).vformat(
                    it
                )[: -(len(codegen.OPENLLM_MODEL_ID) + 3)]
                + "\n"
            )

    script = "".join(service_contents)

    if DEBUG:
        logger.info("Generated script:\n%s", script)

    bento._fs.writetext(service_fs_path, script)

    return bento.save()


@inject
def build(
    model_name: str,
    *,
    model_id: str | None = None,
    model_version: str | None = None,
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    bettertransformer: bool | None = None,
    adapter_map: dict[str, str | None] | None = None,
    build_ctx: str | None = None,
    extra_dependencies: tuple[str, ...] | None = None,
    workers_per_resource: int | float | None = None,
    overwrite_existing_bento: bool = False,
    bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> bentoml.Bento:
    """Package a LLM into a Bento.

    The LLM will be built into a BentoService with the following structure:
    if quantize is passed, it will instruct the model to be quantized dynamically during serving time.
    if bettertransformer is passed, it will instruct the model to use BetterTransformer during serving time.

    Other parameters including model_name, model_id and attrs will be passed to the LLM class itself.
    """
    args = [sys.executable, "-m", "openllm", "build", model_name, "--machine"]

    if quantize and bettertransformer:
        raise openllm.exceptions.OpenLLMException(
            "'quantize' and 'bettertransformer' are currently mutually exclusive."
        )

    if quantize:
        args.extend(["--quantize", quantize])
    if bettertransformer:
        args.append("--bettertransformer")

    if model_id:
        args.extend(["--model-id", model_id])
    if build_ctx:
        args.extend(["--build-ctx", build_ctx])
    if extra_dependencies:
        args.extend([f"--enable-features={f}" for f in extra_dependencies])
    if workers_per_resource:
        args.extend(["--workers-per-resource", str(workers_per_resource)])
    if overwrite_existing_bento:
        args.append("--overwrite")
    if adapter_map:
        args.extend([f"--adapter-id={k}{':'+v if v is not None else ''}" for k, v in adapter_map.items()])
    if model_version:
        args.extend(["--model-version", model_version])

    try:
        output = subprocess.check_output(args, env=os.environ.copy(), cwd=build_ctx or os.getcwd())
    except subprocess.CalledProcessError as e:
        logger.error("Exception caught while building %s", model_name, exc_info=e)
        if e.stderr:
            raise openllm.exceptions.OpenLLMException(e.stderr.decode("utf-8")) from None
        raise openllm.exceptions.OpenLLMException(str(e)) from None
    # NOTE: This usually only concern BentoML devs.
    pattern = r"^__tag__:[^:\n]+:[^:\n]+"
    matched = re.search(pattern, output.decode("utf-8").strip(), re.MULTILINE)
    assert matched is not None, f"Failed to find tag from output: {output}"
    _, _, tag = matched.group(0).partition(":")
    return bentoml.get(tag, _bento_store=bento_store)
