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
import collections
import functools
import hashlib
import inspect
import logging
import os
import re
import types
import typing as t
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import attr
import inflection
import orjson
from huggingface_hub import hf_hub_download

import bentoml
import openllm
from bentoml._internal.models import ModelStore
from bentoml._internal.models.model import ModelSignature
from bentoml._internal.types import ModelSignatureDict

from ._configuration import AdapterType
from ._configuration import FineTuneConfig
from .exceptions import ForbiddenAttributeError
from .exceptions import GpuNotAvailableError
from .utils import DEBUG
from .utils import ENV_VARS_TRUE_VALUES
from .utils import EnvVarMixin
from .utils import LazyLoader
from .utils import ReprMixin
from .utils import bentoml_cattr
from .utils import first_not_none
from .utils import get_debug_mode
from .utils import in_docker
from .utils import is_bitsandbytes_available
from .utils import is_peft_available
from .utils import is_torch_available
from .utils import is_transformers_supports_kbit
from .utils import non_intrusive_setattr
from .utils import normalize_attrs_to_model_tokenizer_pair
from .utils import pkg
from .utils import requires_dependencies
from .utils import resolve_filepath
from .utils import validate_is_path


# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if hasattr(t, "get_overloads"):
    from typing import overload
else:
    from typing_extensions import overload

if t.TYPE_CHECKING:
    import peft
    import torch

    import transformers
    from bentoml._internal.runner.strategy import Strategy

    from ._types import AdaptersMapping
    from ._types import LiteralRuntime
    from ._types import LLMInitAttrs
    from ._types import LLMRunnable as LLMRunnable
    from ._types import LLMRunner
    from ._types import ModelProtocol
    from ._types import PeftAdapterOutput
    from ._types import TokenizerProtocol
    from .utils.representation import ReprArgs

    DictStrAny = dict[str, t.Any]
    TupleAny = tuple[t.Any, ...]
    ListAny = list[t.Any]
    UserDictAny = collections.UserDict[str, t.Any]
else:
    DictStrAny = dict
    TupleAny = tuple
    ListAny = list
    UserDictAny = collections.UserDict
    LLMRunnable = bentoml.Runnable
    LLMRunner = bentoml.Runner
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")
    peft = LazyLoader("peft", globals(), "peft")

logger = logging.getLogger(__name__)


def normalise_model_name(name: str) -> str:
    return os.path.basename(resolve_filepath(name)) if validate_is_path(name) else re.sub("[^a-zA-Z0-9]+", "-", name)


def make_tag(
    model_id: str,
    model_version: str | None = None,
    trust_remote_code: bool = False,
    implementation: LiteralRuntime = "pt",
    quiet: bool = False,
) -> bentoml.Tag:
    """Generate a ``bentoml.Tag`` from a given transformers model name.

    Note that this depends on your model to have a config class available.

    Args:
        model_id: The transformers model name or path to load the model from.
        model_version: Optional model version to be saved with this tag. Default to None.
                       If model_id is a custom path, then the version would be the hash of the last modified
                       time from given ``model_id``.
        trust_remote_code: Whether to trust the remote code. Defaults to False.
        model_version: Optional model version to be saved with this tag.
        implementation: Given implementation for said LLM. One of t.Literal['pt', 'tf', 'flax']
        quiet: Whether to show warning logs. Default to 'False'

    Returns:
        A tuple of ``bentoml.Tag`` and a dict of unused kwargs.
    """
    model_name = normalise_model_name(model_id)

    if validate_is_path(model_id):
        model_id = resolve_filepath(model_id)
        # special cases, if it is the model store, then we return the tags
        # this will happens within the container, where we use the relative path
        if in_docker() and os.getenv("BENTO_PATH") is not None:
            _store = ModelStore(Path(model_id).parent.parent)
            tag = _store.list()[0].tag
            model_version = tag.version
            model_name = tag.name
        elif model_version is None:
            if not quiet:
                logger.warning(
                    "Given 'model_id=%s' is a path, and 'model_version' is not passed. OpenLLM will generate the version based on the last modified time of this given directory.",
                    model_id,
                )
            model_version = generate_hash_from_file(model_id)
    else:
        config = t.cast(
            "transformers.PretrainedConfig",
            transformers.AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                revision=first_not_none(model_version, default="main"),
            ),
        )

        model_version = getattr(config, "_commit_hash", None)
        if model_version is None:
            raise ValueError(
                f"Internal errors when parsing config for pretrained {model_id} ('commit_hash' not found)"
            )

    if in_docker() and os.getenv("BENTO_PATH") is not None:
        logger.debug("The model will be loaded as relative path within BentoContainer.")
    else:
        logger.debug(
            "'model_id=%s' will use 'model_version=%s'. The full tag to be saved under model store: '%s-%s:%s'",
            model_id,
            model_version,
            implementation,
            model_name,
            model_version,
        )

    return bentoml.Tag.from_taglike(f"{implementation}-{model_name}:{model_version}".strip())


@functools.lru_cache(maxsize=128)
def generate_hash_from_file(f: str, algorithm: t.Literal["md5", "sha1"] = "sha1") -> str:
    """Generate a hash from given file's modification time.

    Args:
        f: The file to generate the hash from.
        algorithm: The hashing algorithm to use. Defaults to 'sha1' (similar to how Git generate its commit hash.)

    Returns:
        The generated hash.
    """
    modification_time = str(os.path.getmtime(resolve_filepath(f)))
    hashed = getattr(hashlib, algorithm)(modification_time.encode())
    return hashed.hexdigest()


# the below is similar to peft.utils.other.CONFIG_NAME
PEFT_CONFIG_NAME = "adapter_config.json"


def resolve_peft_config_type(adapter_map: dict[str, str | None] | None):
    """Resolve the type of the PeftConfig given the adapter_map.

    This is similar to how PeftConfig resolve its config type.

    Args:
        adapter_map: The given mapping from either SDK or CLI. See CLI docs for more information.
    """
    if adapter_map is None:
        return

    resolved: AdaptersMapping = {}
    _has_set_default = False
    for path_or_adapter_id, name in adapter_map.items():
        if _has_set_default:
            raise ValueError("Only one adapter can be set as default.")
        if os.path.isfile(os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)):
            config_file = os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(path_or_adapter_id, PEFT_CONFIG_NAME)
            except Exception:
                raise ValueError(f"Can't find '{PEFT_CONFIG_NAME}' at '{path_or_adapter_id}'")
        with open(config_file, "r") as file:
            resolved_config = orjson.loads(file.read())
        # all peft_type should be available in PEFT_CONFIG_NAME
        _peft_type: AdapterType = resolved_config["peft_type"].lower()
        if _peft_type not in resolved:
            resolved[_peft_type] = ()
        resolved[_peft_type] += ((path_or_adapter_id, name, resolved_config),)
        if name == "default":
            _has_set_default = True
    return resolved


_reserved_namespace = {"config_class", "model", "tokenizer", "import_kwargs"}


_M = t.TypeVar("_M", bound="ModelProtocol[t.Any]")
_T = t.TypeVar("_T", bound="TokenizerProtocol[t.Any]")


def _default_post_init(self: LLM[t.Any, t.Any]):
    self.device = None

    if self.__llm_implementation__ == "pt" and is_torch_available():
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLMInterface(ABC, t.Generic[_M, _T]):
    """This defines the loose contract for all openllm.LLM implementations."""

    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]] | None:
        """The default import kwargs to used when importing the model.

        This will be passed into 'openllm.LLM.import_model'.
        It returns two dictionaries: one for model kwargs and one for tokenizer kwargs.
        """
        return

    @abstractmethod
    def generate(self, prompt: str, **preprocess_generate_kwds: t.Any) -> t.Any:
        """The main function implementation for generating from given prompt.  It takes the prompt.

        and the generation_kwargs from 'self.sanitize_parameters' and then
        pass it to 'self.model.generate'.
        """
        raise NotImplementedError

    def generate_one(
        self,
        prompt: str,
        stop: list[str],
        **preprocess_generate_kwds: t.Any,
    ) -> list[dict[t.Literal["generated_text"], str]]:
        """The entrypoint for generating one prompt.

        This provides additional stop tokens for generating per token level.
        This is useful when running with agents, or initial streaming support.
        """
        raise NotImplementedError

    def generate_iterator(self, prompt: str, **attrs: t.Any) -> t.Iterator[t.Any]:
        """An iterator version of generate function."""
        raise NotImplementedError(
            "Currently generate_iterator requires SSE (Server-side events) support, which is not yet implemented."
        )

    def sanitize_parameters(self, prompt: str, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        """This handler will sanitize all attrs and setup prompt text.

        It takes a prompt that is given by the user, attrs that can be parsed with the prompt.

        Returns a tuple of three items:
        - The attributes dictionary that can be passed into LLMConfig to generate a GenerationConfig
        - The attributes dictionary that will be passed into `self.postprocess_generate`.
        """
        return prompt, attrs, attrs

    def postprocess_generate(self, prompt: str, generation_result: t.Any, **attrs: t.Any) -> t.Any:
        """This handler will postprocess generation results from LLM.generate and then output nicely formatted results (if the LLM decide to do so.).

        You can customize how the output of the LLM looks with this hook. By default, it is a simple echo.

        NOTE: this will be used from the client side.
        """
        return generation_result

    def llm_post_init(self):
        """This function can be implemented if you need to initialized any additional variables that doesn't concern OpenLLM internals."""
        pass

    def import_model(self, *args: t.Any, trust_remote_code: bool, **attrs: t.Any) -> bentoml.Model:
        """This function can be implemented if default import_model doesn't satisfy your needs.

        Note that tokenizer attrs can be accessed via ``llm.llm_parameters``.

        ```python
        _, tokenizer_attrs = llm.llm_parameters
        ```

        By default, `model_decls` and `model_attrs` is already sanitised and concatenated into `args` and `attrs`
        """
        raise NotImplementedError

    def load_model(self, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any) -> t.Any:
        """This function can be implemented to override the default load_model behaviour.

        See falcon for example implementation.
        """
        raise NotImplementedError

    def load_tokenizer(self, tag: bentoml.Tag, **attrs: t.Any) -> t.Any:
        """This function can be implemented to override how to load the tokenizer.

        See falcon for example implementation.
        """
        raise NotImplementedError

    # NOTE: All fields below are attributes that can be accessed by users.
    config_class: type[openllm.LLMConfig]
    """The config class to use for this LLM. If you are creating a custom LLM, you must specify this class."""

    bettertransformer: bool
    """Whether to load this LLM with FasterTransformer enabled. The order of loading is:

    - If pass within `for_model`, `from_pretrained` or `__init__`.
    - If `self.bettertransformer` is set within `llm_post_init`.
    - Finally, if none of the above, default to self.config['bettertransformer']

    > **Note** that if LoRA is enabled, bettertransformer will be disabled.
    """

    device: torch.device | str | int | None
    """The device to be used for this LLM. If the implementation is 'pt', then it will be torch.device, else string."""

    # NOTE: The following will be populated by __init_subclass__, note that these should be immutable.
    __llm_trust_remote_code__: bool
    """This is used to determine during 'import_model' whether to trust remote code or not.

    This works synonymous with `trust_remote_code` kwarg in transformers Auto classes. If not passed,
    then by default fallback to config_class['trust_remote_code']
    """
    __llm_implementation__: LiteralRuntime
    """This is used to determine which implementation that this LLM has.

    Usually, this will inferred from class name, that follows the HuggingFace's naming convention:

    - `OPTForConditionalGeneration` -> `pt`
    - `TFOPTForConditionalGeneration` -> `tf`
    - `FlaxOPTForConditionalGeneration` -> `flax`
    """
    __llm_model__: _M | peft.PeftModel | torch.nn.Module | None
    """A reference to the actual model. Instead of access this directly, you should use `model` property instead."""
    __llm_tokenizer__: _T | None
    """A reference to the actual tokenizer. Instead of access this directly, you should use `tokenizer` property instead."""
    __llm_bentomodel__: bentoml.Model | None
    """A reference to the bentomodel used for this LLM. Instead of access this directly, you should use `_bentomodel` property instead."""
    __llm_adapter_map__: dict[AdapterType, dict[str | t.Literal["default"], tuple[peft.PeftConfig, str]]] | None
    """A reference to the the cached LoRA adapter mapping."""

    __llm_custom_import__: bool
    """Whether this LLM has a custom import_model"""
    __llm_custom_load__: t.Callable[[t.Self, t.Any, t.Any], None] | None
    """A callable that will be called after the model is loaded. This is set when 'load_model' is implemented"""
    __llm_custom_tokenizer__: t.Callable[[t.Self, t.Any], None] | None
    """A callable that will be called after the tokenizer is loaded. This is set when 'load_tokenizer' is implemented"""
    __llm_init_kwargs__: property | None
    """A check if 'import_kwargs' is implemented in subclass."""

    if t.TYPE_CHECKING:

        def __attrs_init__(
            self,
            config: openllm.LLMConfig,
            quantization_config: transformers.BitsAndBytesConfig | None,
            model_id: str,
            runtime: t.Literal["ggml", "transformers"],
            model_decls: TupleAny,
            model_attrs: DictStrAny,
            tokenizer_attrs: DictStrAny,
            tag: bentoml.Tag,
            adapters_mapping: AdaptersMapping,
            model_version: str | None,
            quantize_method: t.Literal["int8", "int4", "gptq"] | None,
            /,
            **attrs: t.Unpack[LLMInitAttrs],
        ) -> None:
            """Generated __attrs_init__ for openllm.LLM."""


@attr.define(slots=True, repr=False)
class LLM(LLMInterface[_M, _T], ReprMixin):
    config: openllm.LLMConfig
    """The config instance to use for this LLM. This will be created based on config_class and available
    when initialising the LLM."""

    quantization_config: transformers.BitsAndBytesConfig | None
    """Quantisation config for quantised model on the fly."""

    _model_id: str
    _runtime: t.Literal["ggml", "transformers"]
    _model_decls: TupleAny
    _model_attrs: DictStrAny
    _tokenizer_attrs: DictStrAny
    _tag: bentoml.Tag
    _adapters_mapping: AdaptersMapping
    _model_version: str
    _quantize_method: t.Literal["int8", "int4", "gptq"] | None

    def __init_subclass__(cls):
        cd = cls.__dict__
        prefix_class_name_config = cls.__name__
        if prefix_class_name_config.startswith("Flax"):
            implementation = "flax"
            prefix_class_name_config = prefix_class_name_config[4:]
        elif prefix_class_name_config.startswith("TF"):
            implementation = "tf"
            prefix_class_name_config = prefix_class_name_config[2:]
        else:
            implementation = "pt"
        cls.__llm_implementation__ = implementation
        config_class = openllm.AutoConfig.infer_class_from_name(prefix_class_name_config)

        if "__openllm_internal__" in cd:
            if "config_class" not in cd:
                cls.config_class = config_class
            else:
                logger.debug("Using config class %s for %s.", cd["config_class"], cls.__name__)
        elif "config_class" not in cd:
            raise RuntimeError("Missing required key 'config_class'. Make sure to define it within the LLM subclass.")

        _custom_import = True
        if cls.import_model is LLMInterface[_M, _T].import_model:
            # using the default import model if no custom import is set
            _custom_import = False
            setattr(cls, "import_model", openllm.serialisation.import_model)
        else:
            import_func = getattr(cls, "import_model")

            def _wrapped_import_model(self: LLM[_M, _T], *decls: t.Any, trust_remote_code: bool, **attrs: t.Any):
                # wrapped around custom init to provide some meta compression
                # for all decls and attrs
                (model_decls, model_attrs), _ = self.llm_parameters

                decls = (*model_decls, *decls)
                attrs = {**model_attrs, **attrs}

                return import_func(self, *decls, trust_remote_code=trust_remote_code, **attrs)

            setattr(cls, "import_model", functools.update_wrapper(_wrapped_import_model, cls.import_model))

        if cls.llm_post_init is LLMInterface[_M, _T].llm_post_init:
            # using the default post init if no custom post init is set
            wrapped_post_init = _default_post_init
        else:
            original_post_init = getattr(cls, "llm_post_init")

            def wrapped_post_init(self: LLM[_M, _T]):
                _default_post_init(self)
                original_post_init(self)

        setattr(cls, "llm_post_init", wrapped_post_init)

        cls.__llm_custom_import__ = _custom_import
        cls.__llm_custom_load__ = None if cls.load_model is LLMInterface[_M, _T].load_model else cls.load_model
        cls.__llm_custom_tokenizer__ = (
            None if cls.load_tokenizer is LLMInterface[_M, _T].load_tokenizer else cls.load_tokenizer
        )
        cls.__llm_init_kwargs__ = (
            None if cls.import_kwargs is LLMInterface[_M, _T].import_kwargs else cls.import_kwargs
        )

        for at in {"bentomodel", "model", "tokenizer", "adapter_map"}:
            setattr(cls, f"__llm_{at}__", None)

        # update docstring for given entrypoint
        for fn in {"generate", "generate_one", "generate_iterator"}:
            original_fn = getattr(cls, fn, getattr(LLMInterface, fn))
            original_fn.__doc__ = (
                original_fn.__doc__
                or f"""\
            '{fn}' implementation {cls.__name__}.

            Note that if LoRA is enabled (via either SDK or CLI), `self.model` will become a `peft.PeftModel`
            The original can then be accessed with 'self.model.get_base_model()'.
            """
            )
            setattr(cls, fn, original_fn)

    # The following is the similar interface to HuggingFace pretrained protocol.
    def save_pretrained(self, save_directory: str | Path, **attrs: t.Any) -> None:
        if isinstance(save_directory, Path):
            save_directory = str(save_directory)
        if self.__llm_model__ is not None and self.bettertransformer:
            from optimum.bettertransformer import BetterTransformer

            self.__llm_model__ = BetterTransformer.reverse(self.__llm_model__)

        openllm.serialisation.save_pretrained(self, save_directory, **attrs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        model_version: str | None = None,
        llm_config: openllm.LLMConfig | None = None,
        *args: t.Any,
        runtime: t.Literal["ggml", "transformers"] | None = None,
        quantize: t.Literal["int8", "int4", "gptq"] | None = None,
        bettertransformer: str | bool | None = None,
        adapter_id: str | None = None,
        adapter_name: str | None = None,
        adapter_map: dict[str, str | None] | None = None,
        quantization_config: transformers.BitsAndBytesConfig | None = None,
        **attrs: t.Any,
    ) -> LLM[_M, _T]:
        """Instantiate a pretrained LLM.

        ``LLM.from_pretrained`` follows the same design principle as HuggingFace's `from_pretrained` method, plus the following:

        ### Optimization options:

        > This is most notable during serving time.

        - quantize: quantize the model with the given quantization method. Currently supported int8, int4 quantization
        - bettertransformer: Apply FasterTransformer to given pretrained weight

        > Currently, the above two options are mutually exclusive.

        ### Adapter options:

        > This is used in conjunction with the fine-tuning features

        - adapter_id: Optional [LoRA](https://arxiv.org/pdf/2106.09685.pdf) pretrained id or local path to apply to said model.
        - adapter_name: Optional name of the adapter to apply to said model. If not provided, it will be handled internally by OpenLLM.
        - adapter_map: optional dictionary of adapter_id to adapter_name. Note that this is mutually exclusive with adapter_id/adapter_name arguments.

        Args:
            model_id: The pretrained model to use. Defaults to None. If None, 'self.default_id' will be used.
                      > **Warning**: If custom path is passed, make sure it contains all available file to construct
                      > ``transformers.PretrainedConfig``, ``transformers.PreTrainedModel``, and ``transformers.PreTrainedTokenizer``.
            model_name: Optional model name to be saved with this LLM. Default to None. It will be inferred automatically from model_id.
                        If model_id is a custom path, it will be the basename of the given path.
            model_version: Optional version for this given model id. Default to None. This is useful for saving from custom path.
                           If set to None, the version will either be the git hash from given pretrained model, or the hash inferred
                           from last modified time of the given directory.
            llm_config: The config to use for this LLM. Defaults to None. If not passed, OpenLLM
                        will use `config_class` to construct default configuration.
            quantize: The quantization to use for this LLM. Defaults to None. Possible values
                      include int8, int4 and gptq.
            runtime: Optional runtime to run this LLM. Default to 'transformers'. 'ggml' supports is working in progress.
            quantization_config: The quantization config (`transformers.BitsAndBytesConfig`) to use. Note that this is mutually exclusive with `quantize`
            bettertransformer: Whether to use BetterTransformer with this model. Defaults to False.
            adapter_id: The [LoRA](https://arxiv.org/pdf/2106.09685.pdf) pretrained id or local path to use for this LLM. Defaults to None.
            adapter_name: The adapter name to use for this LLM. Defaults to None.
            adapter_map: The adapter map to use for this LLM. Defaults to None. Note that this is mutually exclusive with adapter_id/adapter_name arguments.
            *args: The args to be passed to the model.
            **attrs: The kwargs to be passed to the model.
        """
        cfg_cls = cls.config_class
        if model_id is None:
            model_id = first_not_none(
                cfg_cls.__openllm_env__["model_id_value"], default=cfg_cls.__openllm_default_id__
            )
        if runtime is None:
            runtime = cfg_cls.__openllm_runtime__

        # quantization setup
        if quantization_config and quantize:
            raise ValueError(
                """'quantization_config' and 'quantize' are mutually exclusive. Either customise
            your quantization_config or use the 'quantize' argument."""
            )

        # 8 bit configuration
        int8_threshold = attrs.pop("llm_int8_threshhold", 6.0)
        cpu_offloading = attrs.pop("llm_int8_enable_fp32_cpu_offload", False)
        int8_skip_modules: list[str] | None = attrs.pop("llm_int8_skip_modules", None)
        int8_has_fp16_weight = attrs.pop("llm_int8_has_fp16_weight", False)
        # 4 bit configuration
        int4_compute_dtype = attrs.pop("bnb_4bit_compute_dtype", torch.bfloat16)
        int4_quant_type = attrs.pop("bnb_4bit_quant_type", "nf4")
        int4_use_double_quant = attrs.pop("bnb_4bit_use_double_quant", True)

        # NOTE: Quantization setup
        # quantize is a openllm.LLM feature, where we can quantize the model
        # with bitsandbytes or quantization aware training.
        if quantization_config is None and quantize is not None:
            if not is_bitsandbytes_available():
                raise RuntimeError(
                    "Quantization requires bitsandbytes to be installed. Make "
                    "sure to install OpenLLM with 'pip install \"openllm[fine-tune]\"'"
                )
            logger.debug(
                "'quantize' is not None. %s will use a default 'quantization_config' for %s. "
                "If you want to customise the quantization config, make sure to pass your "
                "own 'quantization_config'",
                cls.__name__,
                quantize,
            )
            if quantize == "int8":
                if int8_skip_modules is None:
                    int8_skip_modules = []
                if "lm_head" not in int8_skip_modules and cls.config_class.__openllm_model_type__ == "causal_lm":
                    logger.debug("Skipping 'lm_head' for quantization for %s", cls.__name__)
                    int8_skip_modules.append("lm_head")
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=cpu_offloading,
                    llm_int8_threshhold=int8_threshold,
                    llm_int8_skip_modules=int8_skip_modules,
                    llm_int8_has_fp16_weight=int8_has_fp16_weight,
                )
            elif quantize == "int4":
                if is_transformers_supports_kbit():
                    quantization_config = transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=int4_compute_dtype,
                        bnb_4bit_quant_type=int4_quant_type,
                        bnb_4bit_use_double_quant=int4_use_double_quant,
                    )
                else:
                    logger.warning(
                        "'quantize' is set to int4, while the current transformers version %s does not support "
                        "k-bit quantization. k-bit quantization is supported since transformers 4.30, therefore "
                        "make sure to install the latest version of transformers either via PyPI or "
                        "from git source: 'pip install git+https://github.com/huggingface/transformers'.",
                        pkg.pkg_version_info("transformers"),
                    )
            elif quantize == "gptq":
                # TODO: support GPTQ loading quantization
                raise NotImplementedError("GPTQ is not supported yet.")
                if model_id is None:
                    raise RuntimeError(
                        "'quantize=%s' requires passing custom path to quantized weights as we are unable to load "
                        "the model on the fly. See https://github.com/qwopqwop200/GPTQ-for-LLaMa for "
                        "instruction on how to quantize '%s' with GPTQ.",
                        quantize,
                        cls.__name__,
                    )
            else:
                raise ValueError(f"'quantize' must be one of ['int8', 'int4', 'gptq'], got {quantize} instead.")

        # NOTE: Fine-tuning setup
        if adapter_map and adapter_id:
            raise ValueError(
                """'adapter_map' and 'adapter_id' are mutually exclusive. Either provide a
                'adapter_map' ({adapter_id: adapter_name | None, ...}) or use
                the combination of adapter_id/adapter_name arguments.
                """
            )
        if adapter_map is None and adapter_id is not None:
            adapter_map = {adapter_id: adapter_name}

        if adapter_map is not None and not is_peft_available():
            raise RuntimeError(
                "LoRA adapter requires 'peft' to be installed. Make sure to install OpenLLM with 'pip install \"openllm[fine-tune]\"'"
            )

        if llm_config is None:
            llm_config = cls.config_class.model_construct_env(**attrs)
            # The rests of the kwargs that is not used by the config class should be stored into __openllm_extras__.
            attrs = llm_config["extras"]

        try:
            _tag = bentoml.Tag.from_taglike(model_id)
        except ValueError:
            _tag = make_tag(
                model_id,
                model_version=model_version,
                trust_remote_code=cfg_cls.__openllm_trust_remote_code__,
                implementation=cls.__llm_implementation__,
                quiet=True,
            )
        if _tag.version is None:
            raise RuntimeError("Failed to resolve model version.")

        return cls(
            model_id=model_id,
            llm_config=llm_config,
            *args,
            bettertransformer=str(bettertransformer).upper() in ENV_VARS_TRUE_VALUES,
            quantization_config=quantization_config,
            _tag=_tag,
            _model_version=_tag.version,
            _quantize_method=quantize,
            _adapters_mapping=resolve_peft_config_type(adapter_map),
            _runtime=runtime,
            **attrs,
        )

    def __init__(
        self,
        model_id: str,
        llm_config: openllm.LLMConfig,
        *args: t.Any,
        bettertransformer: bool | None = None,
        quantization_config: transformers.BitsAndBytesConfig | None = None,
        _adapters_mapping: dict[AdapterType, tuple[tuple[str | None, str | None, dict[str, t.Any]], ...]]
        | None = None,
        _tag: bentoml.Tag,
        _quantize_method: t.Literal["int8", "int4", "gptq"] | None,
        _runtime: t.Literal["ggml", "transformers"],
        _model_version: str,
        **attrs: t.Any,
    ):
        """Initialize the LLM with given pretrained model.

        > **Warning**
        > To initializing any LLM, you should use `openllm.AutoLLM` or `openllm.LLM.from_pretrained` instead.
        > `__init__` initialization is only for internal use.

        Note:
        - *args to be passed to the model.
        - **attrs will first be parsed to the AutoConfig, then the rest will be parsed to the import_model
        - for tokenizer kwargs, it should be prefixed with _tokenizer_*

        For custom pretrained path, it is recommended to pass in 'model_version' alongside with the path
        to ensure that it won't be loaded multiple times.
        Internally, if a pretrained is given as a HuggingFace repository path , OpenLLM will usethe commit_hash
        to generate the model version.

        For better consistency, we recommend users to also push the fine-tuned model to HuggingFace repository.

        If you need to overwrite the default ``import_model``, implement the following in your subclass:

        ```python
        def import_model(
            self,
            *args: t.Any,
            trust_remote_code: bool,
            **attrs: t.Any,
        ):
            _, tokenizer_attrs = self.llm_parameters

            return bentoml.transformers.save_model(
                tag,
                transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_id, device_map="auto", torch_dtype=torch.bfloat16, **attrs
                ),
                custom_objects={
                    "tokenizer": transformers.AutoTokenizer.from_pretrained(
                        self.model_id, padding_size="left", **tokenizer_attrs
                    )
                },
            )
        ```

        If your import model doesn't require customization, you can simply pass in `import_kwargs`
        at class level that will be then passed into The default `import_model` implementation.
        See ``openllm.DollyV2`` for example.

        ```python
        dolly_v2_runner = openllm.Runner(
            "dolly-v2", _tokenizer_padding_size="left", torch_dtype=torch.bfloat16, device_map="cuda"
        )
        ```

        Note: If you implement your own `import_model`, then `import_kwargs` will be the
        base kwargs. You can still override those via ``openllm.Runner``.

        Note that this tag will be generated based on `self.default_id`.
        passed from the __init__ constructor.

        ``llm_post_init`` can also be implemented if you need to do any additional
        initialization after everything is setup.

        Note: If you need to implement a custom `load_model`, the following is an example from Falcon implementation:

        ```python
        def load_model(self, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any) -> t.Any:
            torch_dtype = attrs.pop("torch_dtype", torch.bfloat16)
            device_map = attrs.pop("device_map", "auto")

            _ref = bentoml.transformers.get(tag)

            model = bentoml.transformers.load_model(_ref, device_map=device_map, torch_dtype=torch_dtype, **attrs)
            return transformers.pipeline("text-generation", model=model, tokenizer=_ref.custom_objects["tokenizer"])
        ```

        Args:
            model_id: The pretrained model to use. Defaults to None. If None, 'self.default_id' will be used.
            llm_config: The config to use for this LLM. Defaults to None. If not passed, OpenLLM
                        will use `config_class` to construct default configuration.
            bettertransformer: Whether to use BetterTransformer with this model. Defaults to False.
            quantization_config: ``transformers.BitsAndBytesConfig`` configuration, or 'gptq' denoting this model to be loaded with GPTQ.
            *args: The args to be passed to the model.
            **attrs: The kwargs to be passed to the model.
        """
        # low_cpu_mem_usage is only available for model
        # this is helpful on system with low memory to avoid OOM
        low_cpu_mem_usage = attrs.pop("low_cpu_mem_usage", True)

        if self.__llm_implementation__ == "pt":
            attrs.update({"low_cpu_mem_usage": low_cpu_mem_usage, "quantization_config": quantization_config})

        model_kwds, tokenizer_kwds = {}, {}
        if self.__llm_init_kwargs__:
            # NOTE: recast here for type safety
            model_kwds, tokenizer_kwds = t.cast("tuple[DictStrAny, DictStrAny]", self.__llm_init_kwargs__)
            logger.debug(
                '\'%s\' default kwargs for model: "%s", tokenizer: "%s"',
                self.__class__.__name__,
                model_kwds,
                tokenizer_kwds,
            )

        # parsing tokenizer and model kwargs, as the hierachy is param pass > default
        normalized_model_kwds, normalized_tokenizer_kwds = normalize_attrs_to_model_tokenizer_pair(**attrs)
        # NOTE: Save the args and kwargs for latter load

        self.__attrs_init__(
            llm_config,
            quantization_config,
            model_id,
            _runtime,
            args,
            {**model_kwds, **normalized_model_kwds},
            {**tokenizer_kwds, **normalized_tokenizer_kwds},
            _tag,
            _adapters_mapping,
            _model_version,
            _quantize_method,
        )
        # handle trust_remote_code
        self.__llm_trust_remote_code__ = self._model_attrs.pop("trust_remote_code", self.config["trust_remote_code"])

        self.llm_post_init()

        # we set it here so that we allow subclass to overwrite bettertransformer in llm_post_init
        if bettertransformer is True:
            logger.debug("Using %r with BetterTransformer", self)
            self.bettertransformer = bettertransformer
        else:
            non_intrusive_setattr(self, "bettertransformer", self.config["bettertransformer"])
        # If lora is passed, the disable bettertransformer
        if _adapters_mapping and self.bettertransformer is True:
            logger.debug("LoRA is visible for %s, disabling BetterTransformer", self)
            self.bettertransformer = False

    def __setattr__(self, attr: str, value: t.Any):
        if attr in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"{attr} should not be set during runtime "
                f"as these value will be reflected during runtime. "
                f"Instead, you can create a custom LLM subclass {self.__class__.__name__}."
            )

        super().__setattr__(attr, value)

    @property
    def adapters_mapping(self) -> AdaptersMapping:
        return self._adapters_mapping

    @adapters_mapping.setter
    def adapters_mapping(self, value: AdaptersMapping):
        self._adapters_mapping = value

    @property
    def __repr_keys__(self) -> set[str]:
        return {"model_id", "runner_name", "config", "adapters_mapping", "runtime", "tag"}

    def __repr_args__(self) -> ReprArgs:
        for k in self.__repr_keys__:
            if k == "config":
                yield k, self.config.model_dump(flatten=True)
            else:
                yield k, getattr(self, k)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def runtime(self) -> t.Literal["ggml", "transformers"]:
        return self._runtime

    @property
    def runner_name(self) -> str:
        return f"llm-{self.config['start_name']}-runner"

    # NOTE: The section below defines a loose contract with langchain's LLM interface.
    @property
    def llm_type(self) -> str:
        return normalise_model_name(self._model_id)

    @property
    def identifying_params(self) -> dict[str, t.Any]:
        return {
            "configuration": self.config.model_dump_json().decode(),
            "model_ids": orjson.dumps(self.config["model_ids"]).decode(),
        }

    @property
    def llm_parameters(self) -> tuple[tuple[tuple[t.Any, ...], dict[str, t.Any]], dict[str, t.Any]]:
        """Returning the processed model and tokenizer parameters.

        These can then be used with 'import_model' or any other place that requires loading model and tokenizer.

        See 'openllm.cli.download_models' for example usage.

        Returns:
            ``tuple``: It returns a tuple of (model_args, model_kwargs) & tokenizer_kwargs
        """
        return (self._model_decls, self._model_attrs), self._tokenizer_attrs

    @property
    def tag(self) -> bentoml.Tag:
        return self._tag

    def ensure_model_id_exists(self) -> bentoml.Model:
        """This utility function will download the model if it doesn't exist yet.

        Make sure to call this function if 'ensure_available' is not set during
        Auto LLM initialisation.

        The equivalent for ``openllm.Runner`` is ``openllm.Runner.download_model``.
        """
        additional_args = ["--machine"]
        if not get_debug_mode():
            additional_args.append("--quiet")
        return openllm.import_model(
            self.config["start_name"],
            model_id=self.model_id,
            model_version=self._model_version,
            runtime=self.runtime,
            implementation=self.__llm_implementation__,
            quantize=self._quantize_method,
            additional_args=additional_args,
        )

    @property
    def _bentomodel(self) -> bentoml.Model:
        if self.__llm_bentomodel__ is None:
            self.__llm_bentomodel__ = openllm.serialisation.get(self)
        return self.__llm_bentomodel__

    @property
    def model(self) -> _M:
        """The model to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        # Run check for GPU
        if self.config["requires_gpu"] and len(openllm.utils.gpu_count()) < 1:
            raise GpuNotAvailableError(f"{self} only supports running with GPU (None available).") from None

        if self.__llm_model__ is None:
            self.__llm_model__ = t.cast(
                _M, openllm.serialisation.load_model(self, *self._model_decls, **self._model_attrs)
            )
        return t.cast(_M, self.__llm_model__)

    @property
    def tokenizer(self) -> _T:
        """The tokenizer to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        if self.__llm_tokenizer__ is None:
            self.__llm_tokenizer__ = t.cast(_T, openllm.serialisation.load_tokenizer(self))
        return self.__llm_tokenizer__

    def _transpose_adapter_mapping(
        self,
        inference_mode: bool = True,
        use_cache: bool = True,
    ) -> dict[AdapterType, dict[str | t.Literal["default"], tuple[peft.PeftConfig, str]]]:
        if self._adapters_mapping is None:
            raise ValueError("LoRA mapping is not set up correctly.")

        if not use_cache:
            logger.debug(
                "'use_cache' is set to False. This means the adapter mapping resolution will not be cached. This should only be used during training."
            )

        if self.__llm_adapter_map__ is not None and use_cache:
            # early out if we already serialized everything.
            return self.__llm_adapter_map__

        adapter_map: dict[AdapterType, dict[str | t.Literal["default"], tuple[peft.PeftConfig, str]]] = {}
        # this is a temporary check to accept the first option name as 'default'
        # then we will raise Error when the optional_name is set to None in next iteration.
        _converted_first_none = False
        for _adapter_type, _adapter_tuple in self._adapters_mapping.items():
            if _adapter_type not in adapter_map:
                adapter_map[_adapter_type] = {}
            default_config = self.config["fine_tune_strategies"].get(
                _adapter_type, FineTuneConfig(adapter_type=_adapter_type, llm_config_class=self.config_class)
            )
            default_config = default_config.eval() if inference_mode else default_config.train()
            for pretrained_or_peft_id, optional_name, resolved_mapping in _adapter_tuple:
                if not optional_name:
                    if not _converted_first_none:
                        _converted_first_none = True
                        optional_name = "default"  # noqa: PLW2901
                    else:
                        raise ValueError(
                            f"{self.__class__.__name__} doesn't know how to resolve adapter_name None mapping: {pretrained_or_peft_id, resolved_mapping}"
                        )
                if optional_name == "default":
                    adapter_map[_adapter_type][optional_name] = (
                        default_config.with_config(**resolved_mapping).to_peft_config(),
                        pretrained_or_peft_id,
                    )
                else:
                    adapter_map[_adapter_type][optional_name] = (
                        FineTuneConfig(
                            adapter_type=_adapter_type,
                            adapter_config=resolved_mapping,
                            inference_mode=inference_mode,
                            llm_config_class=self.config_class,
                        ).to_peft_config(),
                        pretrained_or_peft_id,
                    )

        if self.__llm_adapter_map__ is None and use_cache:
            self.__llm_adapter_map__ = adapter_map

            return self.__llm_adapter_map__

        return adapter_map

    @requires_dependencies("peft", extra="fine-tune")
    def prepare_for_training(self, adapter_type: AdapterType = "lora", **attrs: t.Any) -> tuple[_M, _T]:
        if pkg.pkg_version_info("peft")[:2] >= (0, 4):
            from peft import prepare_model_for_kbit_training
        else:
            from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training

        peft_config = (
            self.config["fine_tune_strategies"]
            .get(adapter_type, FineTuneConfig(adapter_type=adapter_type, llm_config_class=self.config_class))
            .train()
            .with_config(**attrs)
            .to_peft_config()
        )
        wrapped_peft = peft.get_peft_model(prepare_model_for_kbit_training(self.model), peft_config)
        if DEBUG:
            wrapped_peft.print_trainable_parameters()
        return wrapped_peft, self.tokenizer

    @requires_dependencies("peft", extra="fine-tune")
    def apply_adapter(
        self,
        inference_mode: bool = True,
        adapter_type: AdapterType = "lora",
        load_adapters: t.Literal["all"] | list[str] | None = None,
        use_cache: bool = True,
    ) -> peft.PeftModel | _M | torch.nn.Module:
        """Apply given LoRA mapping to the model.

        Note that the base model can still be accessed via self.model.get_base_model().
        """
        assert self.model, "Internal error: Model is not loaded correctly."  # noqa: S101
        assert self.__llm_model__ is not None  # noqa: S101

        # early out if _adapters_mapping is empty or it is already wrapped
        # with peft.
        if not self._adapters_mapping:
            logger.debug("No adapter mapping is found. Skip applying adapter.")
            return self.__llm_model__

        _mapping = self._transpose_adapter_mapping(inference_mode=inference_mode, use_cache=use_cache)
        if adapter_type not in _mapping:
            raise ValueError(
                f"Given adapter type {adapter_type} is not supported. Please choose from {list(_mapping.keys())}"
            )
        adapter_mapping = _mapping[adapter_type]
        default_config, peft_model_id = adapter_mapping.pop("default", None)
        if default_config is None:
            raise ValueError(
                "There is no 'default' mapping. Please check the adapter mapping and report this bug to the OpenLLM team."
            )

        # the below shared similar logics with `get_peft_model`
        # TODO: Support PromptLearningConfig
        if default_config.task_type not in peft.MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not isinstance(
            default_config, peft.PromptLearningConfig
        ):
            logger.debug(
                "Given task type '%s' is not supported by peft. This means it can be a custom PeftModel implementation. Make sure the adapter is loaded manually before running inference.",
                default_config.task_type,
            )
            self.__llm_model__ = peft.PeftModel(self.__llm_model__, default_config)
        else:
            # this is not ideal to serialize like this, wait until https://github.com/huggingface/peft/pull/612
            # is merged
            peft_class = peft.MODEL_TYPE_TO_PEFT_MODEL_MAPPING[default_config.task_type]
            if t.cast("str | None", default_config.base_model_name_or_path) is not None:
                kwargs: dict[str, t.Any] = {"is_trainable": not inference_mode}
                if "config" in inspect.signature(peft_class.from_pretrained).parameters:
                    kwargs["config"] = default_config
                else:
                    kwargs.update(dict(default_config.to_dict().items()))
                self.__llm_model__ = peft_class.from_pretrained(self.__llm_model__, peft_model_id, **kwargs)
            else:
                # in this case, the given base_model_name_or_path is None. This will be hit during training
                self.__llm_model__ = peft_class(self.__llm_model__, default_config)

        # now we loop through the rest with add_adapter
        if len(adapter_mapping) > 0:
            for adapter_name, _peft_config in adapter_mapping.items():
                self.__llm_model__.add_adapter(adapter_name, _peft_config)

            # optionally load adapters. In case of multiple adapters, or on Runner,
            # we will need to set load_adapters='all'
            if load_adapters is not None:
                adapters_to_load = adapter_mapping.keys() if load_adapters == "all" else load_adapters
                for adapter_name in adapters_to_load:
                    _peft_config, _peft_model_id = adapter_mapping[adapter_name]
                    self.__llm_model__.load_adapter(
                        _peft_model_id,
                        adapter_name=adapter_name,
                        is_trainable=not inference_mode,
                        **dict(_peft_config.to_dict()),
                    )

        return self.__llm_model__

    # order of these fields matter here, make sure to sync it with
    # openllm.models.auto.factory._BaseAutoLLMClass.for_model
    def to_runner(
        self,
        models: list[bentoml.Model] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, ModelSignatureDict | ModelSignature] | None = None,
        scheduling_strategy: type[Strategy] | None = None,
    ) -> LLMRunner:
        """Convert this LLM into a Runner.

        Args:
            models: Any additional ``bentoml.Model`` to be included in this given models.
                    By default, this will be determined from the model_name.
            max_batch_size: The maximum batch size for the runner.
            max_latency_ms: The maximum latency for the runner.
            method_configs: The method configs for the runner.
            strategy: The strategy to use for this runner.
            embedded: Whether to run this runner in embedded mode.
            scheduling_strategy: Whether to create a custom scheduling strategy for this Runner.

        Returns:
            A generated LLMRunner for this LLM.

        NOTE: There are some difference between bentoml.models.get().to_runner() and LLM.to_runner(): 'name'.
        - 'name': will be generated by OpenLLM, hence users don't shouldn't worry about this.
            The generated name will be 'llm-<model-start-name>-runner' (ex: llm-dolly-v2-runner, llm-chatglm-runner)
        - 'embedded': Will be disabled by default. There is no reason to run LLM in embedded mode.
        """
        models = models if models is not None else []

        try:
            models.append(self._bentomodel)
        except bentoml.exceptions.NotFound:
            models.append(openllm.serialisation.get(self, auto_import=True))

        if scheduling_strategy is None:
            from ._strategies import CascadingResourceStrategy

            scheduling_strategy = CascadingResourceStrategy
        else:
            logger.debug("Using custom scheduling strategy: %s", scheduling_strategy)

        generate_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=False))
        generate_iterator_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=True))
        if method_configs is None:
            method_configs = {
                "generate": generate_sig,
                "generate_one": generate_sig,
                "generate_iterator": generate_iterator_sig,
            }
        else:
            signatures = ModelSignature.convert_signatures_dict(method_configs)
            generate_sig = first_not_none(signatures.get("generate"), default=generate_sig)
            generate_iterator_sig = first_not_none(signatures.get("generate_iterator"), default=generate_iterator_sig)

        class _Runnable(bentoml.Runnable):
            SUPPORTED_RESOURCES = ("nvidia.com/gpu", "amd.com/gpu", "cpu")
            SUPPORTS_CPU_MULTI_THREADING = True

            def __init__(__self: _Runnable):
                # NOTE: The side effect of this line
                # is that it will load the imported model during
                # runner startup. So don't remove it!!
                __self.model = self.model  # keep a loaded reference
                if self.adapters_mapping is not None:
                    logger.info("Applying LoRA to %s...", self.runner_name)
                    self.apply_adapter(inference_mode=True, load_adapters="all")

            @bentoml.Runnable.method(batchable=False)
            def set_adapter(__self, adapter_name: str) -> dict[t.Literal["success", "error_msg"], bool | str]:
                if not is_peft_available():
                    return {
                        "success": False,
                        "error_msg": "peft is not available. Make sure to install: 'pip install \"openllm[fine-tune]\"'",
                    }
                if self.__llm_adapter_map__ is None:
                    return {
                        "success": False,
                        "error_msg": "No adapters available for current running server.",
                    }
                if not isinstance(self.model, peft.PeftModel):
                    return {"success": False, "error_msg": "Model is not a PeftModel"}
                try:
                    self.model.set_adapter(adapter_name)
                    return {"success": True, "error_msg": ""}
                except ValueError:
                    logger.info("Adapter %s not found", adapter_name)
                    return {
                        "success": False,
                        "error_msg": f"Adapter {adapter_name} not found. Available adapters: {list(self.model.peft_config)}",
                    }

            @bentoml.Runnable.method(
                batchable=generate_sig.batchable,
                batch_dim=generate_sig.batch_dim,
                input_spec=generate_sig.input_spec,
                output_spec=generate_sig.output_spec,
            )
            def __call__(__self, prompt: str, **attrs: t.Any) -> list[t.Any]:
                return self.generate(prompt, **attrs)

            @bentoml.Runnable.method(
                batchable=generate_sig.batchable,
                batch_dim=generate_sig.batch_dim,
                input_spec=generate_sig.input_spec,
                output_spec=generate_sig.output_spec,
            )
            def generate(__self, prompt: str, **attrs: t.Any) -> list[t.Any]:
                return self.generate(prompt, **attrs)

            @bentoml.Runnable.method(
                batchable=generate_sig.batchable,
                batch_dim=generate_sig.batch_dim,
                input_spec=generate_sig.input_spec,
                output_spec=generate_sig.output_spec,
            )
            def generate_one(
                __self, prompt: str, stop: list[str], **attrs: t.Any
            ) -> list[dict[t.Literal["generated_text"], str]]:
                return self.generate_one(prompt, stop, **attrs)

            @bentoml.Runnable.method(
                batchable=generate_iterator_sig.batchable,
                batch_dim=generate_iterator_sig.batch_dim,
                input_spec=generate_iterator_sig.input_spec,
                output_spec=generate_iterator_sig.output_spec,
            )
            def generate_iterator(__self, prompt: str, **attrs: t.Any) -> t.Generator[t.Any, None, None]:
                yield self.generate_iterator(prompt, **attrs)

        def available_adapters(__self: LLMRunner) -> PeftAdapterOutput:
            if not is_peft_available():
                return {
                    "success": False,
                    "result": {},
                    "error_msg": "peft is not available. Make sure to install: 'pip install \"openllm[fine-tune]\"'",
                }
            if self.__llm_adapter_map__ is None:
                return {
                    "success": False,
                    "result": {},
                    "error_msg": "No adapters available for current running server.",
                }
            if not isinstance(__self.model, peft.PeftModel):
                return {"success": False, "result": {}, "error_msg": "Model is not a PeftModel"}
            return {"success": True, "result": __self.model.peft_config, "error_msg": ""}

        def _wrapped_generate_run(__self: LLMRunner, prompt: str, **kwargs: t.Any) -> t.Any:
            """Wrapper for runner.generate.run() to handle the prompt and postprocessing.

            This will be used for LangChain API.

            Usage:
            ```python
            runner = openllm.Runner("dolly-v2", init_local=True)
            runner("What is the meaning of life?")
            ```
            """
            prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **kwargs)
            generated_result = __self.generate.run(prompt, **generate_kwargs)
            return self.postprocess_generate(prompt, generated_result, **postprocess_kwargs)

        def _wrapped_repr_keys(_: LLMRunner) -> set[str]:
            return {"config", "llm_type", "runner_methods", "runtime", "llm_tag"}

        def _wrapped_repr_args(__self: LLMRunner) -> ReprArgs:
            yield "runner_methods", {
                method.name: {
                    "batchable": method.config.batchable,
                    "batch_dim": method.config.batch_dim if method.config.batchable else None,
                }
                for method in __self.runner_methods
            }
            yield "config", self.config.model_dump(flatten=True)
            yield "llm_type", __self.llm_type
            yield "runtime", self.runtime
            yield "llm_tag", self.tag

        # NOTE: returning the two langchain API's to the runner
        return types.new_class(
            inflection.camelize(self.config["model_name"]) + "Runner",
            (bentoml.Runner,),
            exec_body=lambda ns: ns.update(
                {
                    "llm_type": self.llm_type,
                    "identifying_params": self.identifying_params,
                    "llm_tag": self.tag,
                    "llm": self,  # NOTE: self reference to LLM
                    "config": self.config,
                    "peft_adapters": property(fget=available_adapters),
                    "download_model": self.ensure_model_id_exists,
                    "__call__": _wrapped_generate_run,
                    "__module__": self.__module__,
                    "__doc__": self.config["env"].start_docstring,
                    "__repr__": ReprMixin.__repr__,
                    "__repr_keys__": property(_wrapped_repr_keys),
                    "__repr_args__": _wrapped_repr_args,
                }
            ),
        )(
            types.new_class(
                inflection.camelize(self.config["model_name"]) + "Runnable",
                (_Runnable,),
                {},
                lambda ns: ns.update(
                    {
                        "SUPPORTED_RESOURCES": ("nvidia.com/gpu", "amd.com/gpu")
                        if self.config["requires_gpu"]
                        else ("nvidia.com/gpu", "amd.com/gpu", "cpu"),
                        "__module__": self.__module__,
                        "__doc__": self.config["env"].start_docstring,
                    }
                ),
            ),
            name=self.runner_name,
            embedded=False,
            models=models,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=bentoml_cattr.unstructure(method_configs),
            scheduling_strategy=scheduling_strategy,
        )

    def predict(self, prompt: str, **attrs: t.Any) -> t.Any:
        """The scikit-compatible API for self(...)."""
        return self.__call__(prompt, **attrs)

    def __call__(self, prompt: str, **attrs: t.Any) -> t.Any:
        """Returns the generation result and format the result.

        First, it runs `self.sanitize_parameters` to sanitize the parameters.
        The the sanitized prompt and kwargs will be pass into self.generate.
        Finally, run self.postprocess_generate to postprocess the generated result.

        This allows users to do the following:

        ```python
        llm = openllm.AutoLLM.for_model("dolly-v2")
        llm("What is the meaning of life?")
        ```
        """
        prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **attrs)
        generated_result = self.generate(prompt, **generate_kwargs)
        return self.postprocess_generate(prompt, generated_result, **postprocess_kwargs)


@overload
def Runner(
    model_name: str,
    *,
    model_id: str | None = None,
    init_local: t.Literal[False, True] = ...,
    **attrs: t.Any,
) -> LLMRunner:
    ...


@overload
def Runner(
    model_name: str,
    *,
    model_id: str = ...,
    models: list[bentoml.Model] | None = ...,
    max_batch_size: int | None = ...,
    max_latency_ms: int | None = ...,
    method_configs: dict[str, ModelSignatureDict | ModelSignature] | None = ...,
    embedded: t.Literal[True, False] = ...,
    scheduling_strategy: type[Strategy] | None = ...,
    **attrs: t.Any,
) -> LLMRunner:
    ...


def Runner(
    model_name: str,
    ensure_available: bool | None = None,
    init_local: bool = False,
    implementation: LiteralRuntime | None = None,
    llm_config: openllm.LLMConfig | None = None,
    **attrs: t.Any,
) -> LLMRunner:
    """Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'.

    The behaviour of ensure_available that is synonymous to `AutoLLM.for_model` depends on `init_local`.
    By default, `ensure_available` is synonymous to `init_local`, meaning on the service when creating
    runner, it won't download the model. So before running your BentoML Service, you should create a `on_startup`
    hook to check download if you don't want to do it manually:

    ```python

    runner = openllm.Runner("dolly-v2")

    @svc.on_startup
    def download():
        runner.llm.ensure_model_id_exists()
    ...
    ```

    if `init_local=True` (For development workflow), it will also enable `ensure_available`.
    Default value of `ensure_available` is None. If set then use that given value, otherwise fallback to the aforementioned behaviour.

    Args:
        model_name: Supported model name from 'openllm models'
        ensure_available: If True, it will download the model if it is not available. If False, it will skip downloading the model.
                          If False, make sure the model is available locally.
        implementation: The given Runner implementation one choose for this Runner. By default, it is retrieved from the enviroment variable
                        of the respected model_name. For example: 'flan-t5' -> "OPENLLM_FLAN_T5_FRAMEWORK"
        llm_config: Optional ``openllm.LLMConfig`` to initialise this ``openllm.LLMRunner``.
        init_local: If True, it will initialize the model locally. This is useful if you want to
                    run the model locally. (Symmetrical to bentoml.Runner.init_local())
        **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs
                behaviour
    """
    if llm_config is not None:
        attrs.update(
            {
                "bettertransformer": llm_config["env"]["bettertransformer_value"],
                "quantize": llm_config["env"]["quantize_value"],
                "runtime": llm_config["env"]["runtime_value"],
            }
        )

    if implementation is None:
        implementation = EnvVarMixin(model_name)["framework_value"]

    runner = openllm.infer_auto_class(implementation).create_runner(
        model_name,
        llm_config=llm_config,
        ensure_available=ensure_available if ensure_available is not None else init_local,
        **attrs,
    )

    if init_local:
        runner.init_local(quiet=True)

    return runner
