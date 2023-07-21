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
from __future__ import annotations
import importlib.metadata
import inspect
import logging
import os
import typing as t
from pathlib import Path

import fs
import fs.copy
import fs.errors
import orjson
from packaging.version import Version
from simple_di import Provide
from simple_di import inject

import bentoml
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelStore

from ..exceptions import OpenLLMException
from ..utils import DEBUG
from ..utils import EnvVarMixin
from ..utils import codegen
from ..utils import device_count
from ..utils import is_flax_available
from ..utils import is_tf_available
from ..utils import is_torch_available
from ..utils import pkg
from ..utils import resolve_user_filepath


if t.TYPE_CHECKING:
    from fs.base import FS

    import openllm
    from bentoml._internal.bento import BentoStore

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


def handle_package_version(package: str, has_dockerfile_template: bool, lower_bound: bool = True):
    version = Version(pkg.get_pkg_version(package))
    if version.is_devrelease:
        if has_dockerfile_template:
            logger.warning(
                "Installed %s has version %s as a dev release. This means you have a custom build of %s with %s. Make sure to use custom dockerfile templates (--dockerfile-template) to setup %s correctly. See https://docs.bentoml.com/en/latest/guides/containerization.html#dockerfile-template for more information.",
                package,
                version,
                package,
                "CUDA support" if "cu" in str(version) else "more features",
                package,
            )
        return package
    return f"{package}>={importlib.metadata.version(package)}" if lower_bound else package


def construct_python_options(
    llm: openllm.LLM[t.Any, t.Any],
    llm_fs: FS,
    has_dockerfile_template: bool,
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
        packages += [f"openllm[{k}]" for k in extra_dependencies]

    req = llm.config["requirements"]
    if req is not None:
        packages.extend(req)

    if str(os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD", False)).lower() == "false":
        packages.append(f"bentoml>={'.'.join([str(i) for i in pkg.pkg_version_info('bentoml')])}")

    env: EnvVarMixin = llm.config["env"]
    framework_envvar = env["framework_value"]
    if framework_envvar == "flax":
        if not is_flax_available():
            raise ValueError(f"Flax is not available, while {env.framework} is set to 'flax'")
        packages.extend(
            [
                handle_package_version("flax", has_dockerfile_template),
                handle_package_version("jax", has_dockerfile_template),
                handle_package_version("jaxlib", has_dockerfile_template),
            ]
        )
    elif framework_envvar == "tf":
        if not is_tf_available():
            raise ValueError(f"TensorFlow is not available, while {env.framework} is set to 'tf'")
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
                pkgver = handle_package_version(candidate, has_dockerfile_template)
                if pkgver == candidate:
                    packages.extend(["tensorflow"])
                else:
                    _tf_version = importlib.metadata.version(candidate)
                    packages.extend([f"tensorflow>={_tf_version}"])
                break
            except importlib.metadata.PackageNotFoundError:
                pass
    else:
        if not is_torch_available():
            raise ValueError("PyTorch is not available. Make sure to have it locally installed.")
        packages.extend([handle_package_version("torch", has_dockerfile_template)])

    wheels: list[str] = []
    built_wheels = build_editable(llm_fs.getsyspath("/"))
    if built_wheels is not None:
        wheels.append(llm_fs.getsyspath(f"/{built_wheels.split('/')[-1]}"))

    return PythonOptions(packages=packages, wheels=wheels, lock_packages=False)


def construct_docker_options(
    llm: openllm.LLM[t.Any, t.Any],
    _: FS,
    workers_per_resource: int | float,
    quantize: t.LiteralString | None,
    bettertransformer: bool | None,
    adapter_map: dict[str, str | None] | None,
    dockerfile_template: str | None,
    runtime: t.Literal["ggml", "transformers"],
    serialisation_format: t.Literal["safetensors", "legacy"],
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
        "OPENLLM_SERIALIZATION": serialisation_format,
        "OPENLLM_ADAPTER_MAP": f"'{orjson.dumps(adapter_map).decode()}'",
        "BENTOML_DEBUG": str(get_debug_mode()),
        "BENTOML_CONFIG_OPTIONS": f"'{_bentoml_config_options}'",
    }

    if adapter_map:
        env_dict["BITSANDBYTES_NOWELCOME"] = os.environ.get("BITSANDBYTES_NOWELCOME", "1")

    # We need to handle None separately here, as env from subprocess doesn't
    # accept None value.
    _env = EnvVarMixin(
        llm.config["model_name"], bettertransformer=bettertransformer, quantize=quantize, runtime=runtime
    )

    if _env.bettertransformer_value is not None:
        env_dict[_env.bettertransformer] = _env.bettertransformer_value
    if _env.quantize_value is not None:
        env_dict[_env.quantize] = _env.quantize_value
    env_dict[_env.runtime] = _env.runtime_value

    return DockerOptions(
        cuda_version="11.8.0",
        env=env_dict,
        system_packages=["git"],
        dockerfile_template=dockerfile_template,
    )


@inject
def create_bento(
    bento_tag: bentoml.Tag,
    llm_fs: FS,
    llm: openllm.LLM[t.Any, t.Any],
    workers_per_resource: str | int | float,
    quantize: t.LiteralString | None,
    bettertransformer: bool | None,
    dockerfile_template: str | None,
    adapter_map: dict[str, str | None] | None = None,
    extra_dependencies: tuple[str, ...] | None = None,
    build_ctx: str | None = None,
    runtime: t.Literal["ggml", "transformers"] = "transformers",
    serialisation_format: t.Literal["safetensors", "legacy"] = "safetensors",
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _model_store: ModelStore = Provide[BentoMLContainer.model_store],
) -> bentoml.Bento:
    framework_envvar = llm.config["env"]["framework_value"]
    labels = dict(llm.identifying_params)
    labels.update({"_type": llm.llm_type, "_framework": framework_envvar, "start_name": llm.config["start_name"]})

    if adapter_map:
        labels.update(adapter_map)

    if isinstance(workers_per_resource, str):
        if workers_per_resource == "round_robin":
            workers_per_resource = 1.0
        elif workers_per_resource == "conserved":
            workers_per_resource = 1.0 if device_count() == 0 else float(1 / device_count())
        else:
            try:
                workers_per_resource = float(workers_per_resource)
            except ValueError:
                raise ValueError(
                    "'workers_per_resource' only accept ['round_robin', 'conserved'] as possible strategies."
                ) from None

    logger.info("Building Bento for '%s'", llm.config["start_name"])

    if adapter_map is not None:
        if build_ctx is None:
            raise ValueError("build_ctx is required when 'adapter_map' is not None")
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
        python=construct_python_options(
            llm,
            llm_fs,
            dockerfile_template is None,
            extra_dependencies,
            adapter_map,
        ),
        docker=construct_docker_options(
            llm,
            llm_fs,
            workers_per_resource,
            quantize,
            bettertransformer,
            adapter_map,
            dockerfile_template,
            runtime,
            serialisation_format,
        ),
    )

    bento = bentoml.Bento.create(
        build_config=build_config,
        version=bento_tag.version,
        build_ctx=llm_fs.getsyspath("/"),
    )

    # Now we have to format the model_id accordingly based on the model_fs
    model_type = bento.info.labels["_type"]
    model_framework = bento.info.labels["_framework"]
    # the models should have the type
    try:
        model_store = ModelStore(bento._fs.opendir("models"))
        model = model_store.get(f"{model_framework}-{model_type}")
    except fs.errors.ResourceNotFound:
        # new behaviour with BentoML models
        model = _model_store.get(f"{model_framework}-{model_type}")
    except bentoml.exceptions.NotFound:
        raise OpenLLMException(f"Failed to find models for {llm.config['start_name']}") from None

    # NOTE: the model_id_path here are only used for setting this environment variable within the container
    # built with for BentoLLM.
    service_fs_path = fs.path.join("src", llm.config["service_name"])
    service_path = bento._fs.getsyspath(service_fs_path)
    with open(service_path, "r") as f:
        service_contents = f.readlines()

    rel_path = f"../models/{model.tag.path()}"

    for it in service_contents:
        if codegen.OPENLLM_MODEL_ID in it:
            service_contents[service_contents.index(it)] = (
                codegen.ModelIdFormatter(rel_path).vformat(it)[: -(len(codegen.OPENLLM_MODEL_ID) + 3)] + "\n"
            )
        if "__bento_name__" in it:
            service_contents[service_contents.index(it)] = it.format(__bento_name__=str(bento.tag))

    script = "".join(service_contents)

    if DEBUG:
        logger.info("Generated script:\n%s", script)

    bento._fs.writetext(service_fs_path, script)

    signatures = inspect.signature(bento.save).parameters
    if "model_store" in signatures:
        return bento.save(bento_store=_bento_store, model_store=_model_store)
    # backward arguments. `model_store` is added recently
    return bento.save(bento_store=_bento_store)
