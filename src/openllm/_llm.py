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

import copy
import functools
import logging
import os
import re
import subprocess
import sys
import types
import typing as t
from abc import ABC, abstractmethod

import attr
import bentoml
import inflection
import orjson
from bentoml._internal.models.model import ModelSignature
from bentoml._internal.types import ModelSignatureDict

import openllm

from .exceptions import ForbiddenAttributeError, OpenLLMException
from .utils import (DEBUG, LazyLoader, ModelEnv, bentoml_cattr, first_not_none,
                    get_debug_mode, is_bitsandbytes_available,
                    is_torch_available, non_intrusive_setattr, pkg)

if t.TYPE_CHECKING:
    import torch
    import transformers
    from bentoml._internal.runner.strategy import Strategy

    from .models.auto.factory import _BaseAutoLLMClass

    class LLMRunner(bentoml.Runner):
        __doc__: str
        __module__: str
        llm: openllm.LLM[t.Any, t.Any]
        config: openllm.LLMConfig
        llm_type: str
        identifying_params: dict[str, t.Any]

        def __call__(self, *args: t.Any, **attrs: t.Any) -> t.Any:
            ...

else:
    LLMRunner = bentoml.Runner
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)

FRAMEWORK_TO_AUTOCLASS_MAPPING = {
    "pt": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    "tf": ("TFAutoModelForCausalLM", "TFAutoModelForSeq2SeqLM"),
    "flax": ("FlaxAutoModelForCausalLM", "FlaxAutoModelForSeq2SeqLM"),
}

TOKENIZER_PREFIX = "_tokenizer_"


def convert_transformers_model_name(name: str | None) -> str:
    if name is None:
        raise ValueError("'name' cannot be None")
    if os.path.exists(os.path.dirname(name)):
        name = os.path.basename(name)
        logger.debug("Given name is a path, only returning the basename %s")
        return name
    return re.sub("[^a-zA-Z0-9]+", "-", name)


def import_model(
    model_id: str,
    tag: bentoml.Tag,
    _model_framework: str,
    *model_args: t.Any,
    tokenizer_kwds: dict[str, t.Any],
    **attrs: t.Any,
):
    """Auto detect model type from given model_id and import it to bentoml's model store.

    For all kwargs, it will be parsed into `transformers.AutoConfig.from_pretrained` first,
    returning all of the unused kwargs.
    The unused kwargs then parsed directly into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
    For all tokenizer kwargs, make sure to prefix it with `_tokenizer_` to avoid confusion.

    Note: Currently, there are only two tasks supported: `text-generation` and `text2text-generation`.

    Refer to Transformers documentation for more information about kwargs.

    Args:
        model_id: Model id to be imported. See `openllm models` for all supported models.
        tag: Tag to be used for the model. This is usually generated for you.
        model_args: Args to be passed into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
        **attrs: Kwargs to be passed into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
    """

    config: transformers.PretrainedConfig = attrs.pop("config", None)
    trust_remote_code = attrs.pop("trust_remote_code", False)

    # this logic below is synonymous to handling `from_pretrained` attrs.
    hub_kwds_names = [
        "cache_dir",
        "force_download",
        "local_files_only",
        "proxies",
        "resume_download",
        "revision",
        "subfolder",
        "use_auth_token",
    ]
    hub_attrs = {k: attrs.pop(k) for k in hub_kwds_names if k in attrs}
    if not isinstance(config, transformers.PretrainedConfig):
        copied_attrs = copy.deepcopy(attrs)
        if copied_attrs.get("torch_dtype", None) == "auto":
            copied_attrs.pop("torch_dtype")
        config, attrs = t.cast(
            "tuple[transformers.PretrainedConfig, dict[str, t.Any]]",
            transformers.AutoConfig.from_pretrained(
                model_id, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_attrs, **copied_attrs
            ),
        )

    # NOTE: `1-2` -> text-generation and text2text-generation
    if type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING:
        idx = 0
    elif type(config) in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
        idx = 1
    else:
        raise OpenLLMException(f"Model type {type(config)} is not supported yet.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=trust_remote_code,
        **hub_attrs,
        **tokenizer_kwds,
    )

    model = getattr(
        transformers,
        FRAMEWORK_TO_AUTOCLASS_MAPPING[_model_framework][idx],
    ).from_pretrained(
        model_id,
        *model_args,
        config=config,
        trust_remote_code=trust_remote_code,
        **hub_attrs,
        **attrs,
    )

    try:
        return bentoml.transformers.save_model(tag, model, custom_objects={"tokenizer": tokenizer})
    finally:
        import gc

        gc.collect()

        # NOTE: We need to free up the cache after importing the model
        # in the case where users first run openllm start without the model
        # available locally.
        if is_torch_available() and torch.cuda.is_available():
            torch.cuda.empty_cache()


_reserved_namespace = {"config_class", "model", "tokenizer", "import_kwargs"}


class LLMInterface(ABC):
    """This defines the loose contract for all openllm.LLM implementations."""

    config_class: type[openllm.LLMConfig]
    """The config class to use for this LLM. If you are creating a custom LLM, you must specify this class."""

    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]] | None:
        """The default import kwargs to used when importing the model.
        This will be passed into 'openllm.LLM.import_model'.
        It returns two dictionaries: one for model kwargs and one for tokenizer kwargs.
        """
        return

    @abstractmethod
    def generate(self, prompt: str, **preprocess_generate_kwds: t.Any) -> t.Any:
        """The main function implementation for generating from given prompt.  It takes the prompt
        and the generation_kwargs from 'self.sanitize_parameters' and then
        pass it to 'self.model.generate'.
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
        """This handler will postprocess generation results from LLM.generate and
        then output nicely formatted results (if the LLM decide to do so.)

        You can customize how the output of the LLM looks with this hook. By default, it is a simple echo.

        NOTE: this will be used from the client side.
        """
        return generation_result

    def llm_post_init(self):
        """This function can be implemented if you need to initialized any additional variables that doesn't
        concern OpenLLM internals.
        """
        pass

    def import_model(
        self, model_id: str, tag: bentoml.Tag, *args: t.Any, tokenizer_kwds: dict[str, t.Any], **attrs: t.Any
    ) -> bentoml.Model:
        """This function can be implemented if default import_model doesn't satisfy your needs."""
        raise NotImplementedError

    def load_model(self, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any) -> t.Any:
        """This function can be implemented to override th default load_model behaviour. See falcon for
        example implementation."""
        raise NotImplementedError


_M = t.TypeVar("_M")
_T = t.TypeVar("_T")


@attr.define(slots=True, repr=False)
class LLM(LLMInterface, t.Generic[_M, _T]):
    if t.TYPE_CHECKING:
        # The following will be populated by metaclass
        __llm_trust_remote_code__: bool
        __llm_implementation__: t.Literal["pt", "tf", "flax"]
        __llm_model__: _M | None
        __llm_tokenizer__: _T | None
        __llm_tag__: bentoml.Tag | None
        __llm_bentomodel__: bentoml.Model | None

        __llm_post_init__: t.Callable[[t.Self], None] | None
        __llm_custom_load__: t.Callable[[t.Self, t.Any, t.Any], None] | None
        __llm_init_kwargs__: property | None

        _model_args: tuple[t.Any, ...]
        _model_attrs: dict[str, t.Any]
        _tokenizer_attrs: dict[str, t.Any]

        bettertransformer: bool

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
                logger.debug(f"Using config class {cd['config_class']} for {cls.__name__}.")
        else:
            if "config_class" not in cd:
                raise RuntimeError(
                    "Missing required key 'config_class'. Make sure to define it within the LLM subclass."
                )

        if cls.import_model is LLMInterface.import_model:
            # using the default import model
            setattr(cls, "import_model", functools.partial(import_model, _model_framework=implementation))
        else:
            logger.debug("Custom 'import_model' will be used when loading modelj %s", cls.__name__)

        cls.__llm_post_init__ = None if cls.llm_post_init is LLMInterface.llm_post_init else cls.llm_post_init
        cls.__llm_custom_load__ = None if cls.load_model is LLMInterface.load_model else cls.load_model
        cls.__llm_init_kwargs__ = None if cls.import_kwargs is LLMInterface.import_kwargs else cls.import_kwargs

        for at in {"bentomodel", "tag", "model", "tokenizer"}:
            setattr(cls, f"__llm_{at}__", None)

    # The following is the similar interface to HuggingFace pretrained protocol.
    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        llm_config: openllm.LLMConfig | None = None,
        *args: t.Any,
        quantize: t.Literal["int8", "int4", "gptq"] | None = None,
        bettertransformer: bool | None = None,
        **attrs: t.Any,
    ) -> LLM[_M, _T]:
        return cls(
            model_id=model_id,
            llm_config=llm_config,
            *args,
            quantize=quantize,
            bettertransformer=bettertransformer,
            **attrs,
        )

    def __init__(
        self,
        model_id: str | None = None,
        llm_config: openllm.LLMConfig | None = None,
        *args: t.Any,
        quantize: t.Literal["int8", "int4", "gptq"] | None = None,
        bettertransformer: bool | None = None,
        **attrs: t.Any,
    ):
        """Initialize the LLM with given pretrained model.

        Note:
        - *args to be passed to the model.
        - **attrs will first be parsed to the AutoConfig, then the rest will be parsed to the import_model
        - for tokenizer kwargs, it should be prefixed with _tokenizer_*

        For custom pretrained path, it is recommended to pass in 'openllm_model_version' alongside with the path
        to ensure that it won't be loaded multiple times.
        Internally, if a pretrained is given as a HuggingFace repository path , OpenLLM will usethe commit_hash
        to generate the model version.

        For better consistency, we recommend users to also push the fine-tuned model to HuggingFace repository.

        If you need to overwrite the default ``import_model``, implement the following in your subclass:

        ```python
        def import_model(
            self,
            model_id: str,
            tag: bentoml.Tag,
            *args: t.Any,
            tokenizer_kwds: dict[str, t.Any],
            **attrs: t.Any,
        ):
            return bentoml.transformers.save_model(
                tag,
                transformers.AutoModelForCausalLM.from_pretrained(
                    model_id, device_map="auto", torch_dtype=torch.bfloat16, **attrs
                ),
                custom_objects={
                    "tokenizer": transformers.AutoTokenizer.from_pretrained(
                        model_id, padding_size="left", **tokenizer_kwds
                    )
                },
            )
        ```

        If your import model doesn't require customization, you can simply pass in `import_kwargs`
        at class level that will be then passed into The default `import_model` implementation.
        See ``openllm.DollyV2`` for example.

        ```python
        dolly_v2_runner = openllm.Runner(
            "dolly-v2", _tokenizer_padding_size="left", torch_dtype=torch.bfloat16, device_map="gpu"
        )
        ```

        Note: If you implement your own `import_model`, then `import_kwargs` will be the
        default kwargs for every load. You can still override those via ``openllm.Runner``.

        Note that this tag will be generated based on `self.default_id` or the given `pretrained` kwds.
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
            quantize: The quantization to use for this LLM. Defaults to None. Possible values
                      include int8, int4 and gptq.
            bettertransformer: Whether to use BetterTransformer with this model. Defaults to False.
            *args: The args to be passed to the model.
            **attrs: The kwargs to be passed to the model.

        The following are optional:
            openllm_model_version: version for this `model_id`. By default, users can ignore this if using pretrained
                                   weights as OpenLLM will use the commit_hash of given model_id.
                                   However, if `model_id` is a path, this argument is recomended to include.
        """

        openllm_model_version = attrs.pop("openllm_model_version", None)

        # low_cpu_mem_usage is only available for model
        # this is helpful on system with low memory to avoid OOM
        low_cpu_mem_usage = attrs.pop("low_cpu_mem_usage", True)

        # quantization setup
        quantization_config = attrs.pop("quantization_config", None)
        # 8 bit configuration
        int8_threshold = attrs.pop("llm_int8_threshhold", 6.0)
        cpu_offloading = attrs.pop("llm_int8_enable_fp32_cpu_offload", False)
        int8_skip_modules: list[str] | None = attrs.pop("llm_int8_skip_modules", None)
        int8_has_fp16_weight = attrs.pop("llm_int8_has_fp16_weight", False)
        # 4 bit configuration
        int4_compute_dtype = attrs.pop("llm_bnb_4bit_compute_dtype", torch.bfloat16)
        int4_quant_type = attrs.pop("llm_bnb_4bit_quant_type", "nf4")
        int4_use_double_quant = attrs.pop("llm_bnb_4bit_use_double_quant", True)

        if llm_config is not None:
            logger.debug("Using given 'llm_config=(%s)' to initialize LLM.", llm_config)
            self.config = llm_config
        else:
            self.config = self.config_class.model_construct_env(**attrs)
            # The rests of the kwargs that is not used by the config class should be stored into __openllm_extras__.
            attrs = self.config["extras"]

        if quantization_config and quantize:
            raise ValueError(
                """'quantization_config' and 'quantize' are mutually exclusive. Either customise
            your quantization_config or use the quantize argument."""
            )
        if quantization_config is None:
            # quantize is a openllm.LLM feature, where we can quantize the model
            # with bitsandbytes or quantization aware training.
            if quantize is not None:
                if not is_bitsandbytes_available():
                    raise RuntimeError(
                        "Quantization requires bitsandbytes to be installed. Make "
                        "sure to install OpenLLM with 'pip install \"openllm[fine-tune]\"'"
                    )
                logger.debug(
                    "'quantize' is not None. %s will use a default 'quantization_config' for %s. "
                    "If you want to customise the quantization config, make sure to pass your "
                    "own 'quantization_config'",
                    self,
                    quantize,
                )
                if quantize == "int8":
                    if int8_skip_modules is None:
                        int8_skip_modules = []
                    if "lm_head" not in int8_skip_modules and self.config["model_type"] == "causal_lm":
                        logger.debug("Skipping 'lm_head' for quantization for %s", self)
                        int8_skip_modules.append("lm_head")
                    quantization_config = transformers.BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=cpu_offloading,
                        llm_int8_threshhold=int8_threshold,
                        llm_int8_skip_modules=int8_skip_modules,
                        llm_int8_has_fp16_weight=int8_has_fp16_weight,
                    )
                elif quantize == "int4":
                    trf_versions = pkg.pkg_version_info("transformers")
                    supports_kbits = trf_versions[:2] >= (4, 30)
                    if supports_kbits:
                        quantization_config = transformers.BitsAndBytesConfig(
                            load_in_4bit=True,
                            llm_bnb_4bit_compute_dtype=int4_compute_dtype,
                            llm_bnb_4bit_quant_type=int4_quant_type,
                            llm_bnb_4bit_use_double_quant=int4_use_double_quant,
                        )
                    else:
                        logger.warning(
                            "'quantize' is set to int4, while the current transformers version %s does not support "
                            "k-bit quantization. k-bit quantization is supported since transformers 4.30, therefore "
                            "make sure to install the latest version of transformers either via PyPI or "
                            "from git source: 'pip install git+https://github.com/huggingface/transformers'.",
                            trf_versions,
                        )
                elif quantize == "gptq":
                    # TODO: support GPTQ loading quantization
                    if model_id is None:
                        raise RuntimeError(
                            "'quantize=%s' requires passing custom path to quantized weights as we are unable to load "
                            "the model on the fly. See https://github.com/qwopqwop200/GPTQ-for-LLaMa for "
                            "instruction on how to quantize '%s' with GPTQ.",
                            quantize,
                            self,
                        )
                    raise NotImplementedError("GPTQ is not supported yet.")
                else:
                    raise ValueError(f"'quantize' must be one of ['int8', 'int4', 'gptq'], got {quantize} instead.")

        if self.__llm_implementation__ == "pt":
            if not self.config["use_pipeline"]:
                attrs["low_cpu_mem_usage"] = low_cpu_mem_usage
            attrs["quantization_config"] = quantization_config

        model_kwds, tokenizer_kwds = {}, {}
        if self.__llm_init_kwargs__:
            if t.TYPE_CHECKING:
                # the above meta value should determine that this LLM has custom kwargs
                assert self.import_kwargs
            model_kwds, tokenizer_kwds = self.import_kwargs
            logger.debug(
                "'%s' default kwargs for model: '%s', tokenizer: '%s'",
                self.__class__.__name__,
                model_kwds,
                tokenizer_kwds,
            )

        if model_id is None:
            model_id = os.environ.get(self.config["env"].model_id, self.config["default_id"])

        # NOTE: This is the actual given path or pretrained weight for this LLM.
        assert model_id is not None
        self._model_id = model_id

        # parsing tokenizer and model kwargs
        tokenizer_kwds.update(
            {k[len(TOKENIZER_PREFIX) :]: v for k, v in attrs.items() if k.startswith(TOKENIZER_PREFIX)}
        )
        model_kwds.update({k: v for k, v in attrs.items() if not k.startswith(TOKENIZER_PREFIX)})

        # handle trust_remote_code
        self.__llm_trust_remote_code__ = model_kwds.pop("trust_remote_code", self.config["trust_remote_code"])

        # NOTE: Save the args and kwargs for latter load
        self._model_args = args
        self._model_attrs = model_kwds
        self._tokenizer_attrs = tokenizer_kwds
        self._openllm_model_version = openllm_model_version

        if self.__llm_post_init__:
            self.llm_post_init()

        # we set it here so that we allow subclass to overwrite bettertransformer in llm_post_init
        if bettertransformer:
            logger.debug("Using %r with BetterTransformer", self)
            self.bettertransformer = bettertransformer
        else:
            non_intrusive_setattr(self, "bettertransformer", self.config["bettertransformer"])

    def __setattr__(self, attr: str, value: t.Any):
        if attr in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"{attr} should not be set during runtime "
                f"as these value will be reflected during runtime. "
                f"Instead, you can create a custom LLM subclass {self.__class__.__name__}."
            )

        super().__setattr__(attr, value)

    def __repr__(self) -> str:
        keys = {"model_id", "runner_name", "llm_type", "config"}
        return f"{self.__class__.__name__}({', '.join(f'{k}={getattr(self, k)!r}' for k in keys)})"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def runner_name(self) -> str:
        return f"llm-{self.config['start_name']}-runner"

    # NOTE: The section below defines a loose contract with langchain's LLM interface.
    @property
    def llm_type(self) -> str:
        return convert_transformers_model_name(self._model_id)

    @property
    def identifying_params(self) -> dict[str, t.Any]:
        return {
            "configuration": self.config.model_dump_json().decode(),
            "model_ids": orjson.dumps(self.config["model_ids"]).decode(),
        }

    @property
    def llm_parameters(self) -> tuple[tuple[tuple[t.Any, ...], dict[str, t.Any]], dict[str, t.Any]]:
        """Returning the processed model and tokenizer parameters to be used with
        'import_model' or any other place that requires loading model and tokenizer.

        See 'openllm.cli.download_models' for example usage.
        It returns a tuple of (model_args, model_kwargs) & tokenizer_kwargs
        """
        return (self._model_args, self._model_attrs), self._tokenizer_attrs

    @staticmethod
    def make_tag(
        model_id: str | None = None,
        trust_remote_code: bool = False,
        openllm_model_version: str | None = None,
        implementation: t.Literal["pt", "flax", "tf"] = "pt",
    ) -> bentoml.Tag:
        """Generate a ``bentoml.Tag`` from a given transformers model name.

        Note that this depends on your model to have a config class available.

        Args:
            model_id: The transformers model name or path to load the model from.
                      If it is a path, then `openllm_model_version` must be passed in as a kwarg.
            trust_remote_code: Whether to trust the remote code. Defaults to False.
            openllm_model_version: Optional model version to be saved with this tag.
            implementation: Given implementation for said LLM. One of t.Literal['pt', 'tf', 'flax']

        Returns:
            A tuple of ``bentoml.Tag`` and a dict of unused kwargs.
        """
        name = convert_transformers_model_name(model_id)

        config = t.cast(
            "transformers.PretrainedConfig",
            transformers.AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            ),
        )

        model_version = getattr(config, "_commit_hash", None)
        if model_version is None:
            if openllm_model_version is not None:
                logger.warning(
                    "Given %s from '%s' doesn't contain a commit hash, and 'openllm_model_version' is not given. "
                    "We will generate the tag without specific version.",
                    config.__class__,
                    model_id,
                )
                model_version = bentoml.Tag.from_taglike(name).make_new_version().version
            else:
                logger.debug("Using %s for '%s' as model version", openllm_model_version, model_id)
                model_version = openllm_model_version

        return bentoml.Tag.from_taglike(f"{implementation}-{name}:{model_version}")

    def ensure_model_id_exists(self) -> bentoml.Model:
        """This utility function will download the model if it doesn't exist yet.
        Make sure to call this function if 'ensure_available' is not set during
        Auto LLM initialisation.
        """
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "openllm",
                "download",
                self.config["start_name"],
                "--model-id",
                self.model_id,
                "--output",
                "porcelain",
            ]
        )
        if DEBUG or get_debug_mode():
            # NOTE: This usually only concern BentoML devs.
            pattern = r"^__tag__:[^:\n]+:[^:\n]+"
            matched = re.search(pattern, output.decode("utf-8").strip(), re.MULTILINE)
            assert matched is not None, f"Failed to find tag from output: {output}"
            _, _, tag = matched.group(0).partition(":")
        else:
            tag = output.strip().decode()

        return bentoml.transformers.get(tag)

    @property
    def _bentomodel(self) -> bentoml.Model:
        if self.__llm_bentomodel__ is None:
            # NOTE: Since PR#28, self.__llm_bentomodel__ changed from
            # ensure_model_id_exists() into just returning the model ref.
            # This is because we want to save a few seconds of loading time,
            # as openllm.Runner and openllm.AutoLLM initialisation is around 700ms
            # before #28.
            # If users want to make sure to have the model downloaded,
            # one should invoke `LLM.ensure_model_id_exists()` manually,
            # or pass `ensure_available=True` into the Auto LLM initialisation.
            self.__llm_bentomodel__ = bentoml.transformers.get(self.tag)
        return self.__llm_bentomodel__

    @property
    def tag(self) -> bentoml.Tag:
        if self.__llm_tag__ is None:
            self.__llm_tag__ = self.make_tag(
                self._model_id,
                self.__llm_trust_remote_code__,
                self._openllm_model_version,
                self.__llm_implementation__,
            )
        return self.__llm_tag__

    @property
    def model(self) -> _M:
        """The model to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        # Run check for GPU
        self.config.check_if_gpu_is_available(implementation=self.__llm_implementation__)

        kwds = self._model_attrs
        kwds["trust_remote_code"] = self.__llm_trust_remote_code__

        is_pipeline = "_pretrained_class" in self._bentomodel.info.metadata
        # differentiate when saving tokenizer or other pretrained type.
        is_pretrained_model = is_pipeline and "_framework" in self._bentomodel.info.metadata

        if self.bettertransformer and is_pipeline and self.config["use_pipeline"]:
            # This is a pipeline, provide a accelerator args
            kwds["accelerator"] = "bettertransformer"

        if self.__llm_model__ is None:
            if self.__llm_custom_load__:
                self.__llm_model__ = self.load_model(self.tag, *self._model_args, **kwds)
            else:
                self.__llm_model__ = self._bentomodel.load_model(*self._model_args, **kwds)

            if (
                self.bettertransformer
                and is_pretrained_model
                and self._bentomodel.info.metadata["_framework"] == "torch"
                and self.config["runtime"] == "transformers"
            ):
                # BetterTransformer is currently only supported on PyTorch.
                from optimum.bettertransformer import BetterTransformer

                self.__llm_model__ = BetterTransformer.transform(self.__llm_model__)
        return t.cast(_M, self.__llm_model__)

    @property
    def tokenizer(self) -> _T:
        """The tokenizer to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        if self.__llm_tokenizer__ is None:
            try:
                self.__llm_tokenizer__ = self._bentomodel.custom_objects["tokenizer"]
            except KeyError:
                # This could happen if users implement their own import_model
                raise openllm.exceptions.OpenLLMException(
                    "Model does not have tokenizer. Make sure to save \
                    the tokenizer within the model via 'custom_objects'.\
                    For example: bentoml.transformers.save_model(..., custom_objects={'tokenizer': tokenizer}))"
                )
        return self.__llm_tokenizer__

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

        NOTE: There are some difference between bentoml.models.get().to_runner() and LLM.to_runner(): 'name'.
        - 'name': will be generated by OpenLLM, hence users don't shouldn't worry about this.
            The generated name will be 'llm-<model-start-name>-runner' (ex: llm-dolly-v2-runner, llm-chatglm-runner)
        - 'embedded': Will be disabled by default. There is no reason to run LLM in embedded mode.
        """
        models = models if models is not None else []
        models.append(self._bentomodel)

        if scheduling_strategy is None:
            from bentoml._internal.runner.strategy import DefaultStrategy

            scheduling_strategy = DefaultStrategy

        generate_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=False))
        generate_iterator_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=True))
        if method_configs is None:
            method_configs = {"generate": generate_sig, "generate_iterator": generate_iterator_sig}
        else:
            signatures = ModelSignature.convert_signatures_dict(method_configs)
            generate_sig = first_not_none(signatures.get("generate"), default=generate_sig)
            generate_iterator_sig = first_not_none(signatures.get("generate_iterator"), default=generate_iterator_sig)

        class _Runnable(bentoml.Runnable):
            SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
            SUPPORTS_CPU_MULTI_THREADING = True

            llm_type: str
            identifying_params: dict[str, t.Any]

            def __init_subclass__(cls, llm_type: str, identifying_params: dict[str, t.Any], **_: t.Any):
                cls.llm_type = llm_type
                cls.identifying_params = identifying_params

            def __init__(__self: _Runnable):
                # NOTE: The side effect of this line
                # is that it will load the imported model during
                # runner creation. So don't remove it!!
                self.model

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
                batchable=generate_iterator_sig.batchable,
                batch_dim=generate_iterator_sig.batch_dim,
                input_spec=generate_iterator_sig.input_spec,
                output_spec=generate_iterator_sig.output_spec,
            )
            def generate_iterator(__self, prompt: str, **attrs: t.Any) -> t.Iterator[t.Any]:
                yield self.generate_iterator(prompt, **attrs)

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

        # NOTE: returning the two langchain API's to the runner
        return types.new_class(
            inflection.camelize(self.config["model_name"]) + "Runner",
            (bentoml.Runner,),
            exec_body=lambda ns: ns.update(
                {
                    "llm_type": self.llm_type,
                    "identifying_params": self.identifying_params,
                    "llm": self,  # NOTE: self reference to LLM
                    "config": self.config,
                    "__call__": _wrapped_generate_run,
                    "__module__": f"openllm.models.{self.config['model_name']}",
                    "__doc__": self.config["env"].start_docstring,
                }
            ),
        )(
            types.new_class(
                inflection.camelize(self.config["model_name"]) + "Runnable",
                (_Runnable,),
                {
                    "SUPPORTED_RESOURCES": ("nvidia.com/gpu", "cpu")
                    if self.config["requires_gpu"]
                    else ("nvidia.com/gpu",),
                    "llm_type": self.llm_type,
                    "identifying_params": self.identifying_params,
                },
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
        """The scikit-compatible API for self(...)"""
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


@t.overload
def Runner(
    model_name: str,
    *,
    model_id: str | None = None,
    init_local: t.Literal[False, True] = ...,
    **attrs: t.Any,
) -> LLMRunner:
    ...


@t.overload
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


def Runner(model_name: str, ensure_available: bool = True, init_local: bool = False, **attrs: t.Any) -> LLMRunner:
    """Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'

    Args:
        model_name: Supported model name from 'openllm models'
        ensure_available: If True, it will ensure the model is available before creating the runner.
                          Set to False for faster creation time. Note that you will need to make sure
                          the model for this 'model_id' is available before calling the runner.
                          One can do this by doing the following:
                          ```python
                          runner = openllm.Runner("dolly-v2", ensure_available=False)
                          runner.llm.ensure_model_id_exists()
                          ```
        init_local: If True, it will initialize the model locally. This is useful if you want to
                    run the model locally. (Symmetrical to bentoml.Runner.init_local())
        **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs
                behaviour
    """
    runner = t.cast(
        "_BaseAutoLLMClass",
        openllm[ModelEnv(model_name)["framework_value"]],  # type: ignore (internal API)
    ).create_runner(model_name, ensure_available=ensure_available, **attrs)

    if init_local:
        runner.init_local(quiet=True)

    return runner
