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
import enum
import functools
import logging
import os
import typing as t
from abc import ABC, ABCMeta, abstractmethod

import bentoml
import inflection

import openllm

from ._configuration import ModelSignature
from .exceptions import ForbiddenAttributeError, OpenLLMException
from .utils import bentoml_cattr

if t.TYPE_CHECKING:
    import transformers
    from bentoml._internal.runner.strategy import Strategy

    from ._types import LLMModel, LLMTokenizer, ModelSignatureDict
else:
    ModelSignatureDict = dict
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)

# NOTE: `1-2` -> text-generation and text2text-generation
_return_tensors_to_framework_map = {
    "pt": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    "tf": ("TFAutoModelForCausalLM", "TFAutoModelForSeq2SeqLM"),
    "flax": ("FlaxAutoModelForCausalLM", "FlaxAutoModelForSeq2SeqLM"),
}


class TypeMeta(enum.EnumMeta):
    def __getitem__(self, key: str) -> enum.Enum:
        # Type safe getters
        normalised = key.replace("-", "_").upper()
        try:
            return self._member_map_[normalised]
        except KeyError:
            raise OpenLLMException(
                f"TaskType '{key}' is not yet supported. Current supported tasks: {set(self._member_map_)}"
            ) from None
        except TypeError:
            raise OpenLLMException(f"getitem key must be a string. Got {type(key)} instead.") from None


@enum.unique
class TaskType(enum.Enum, metaclass=TypeMeta):
    TEXT_GENERATION = enum.auto()
    TEXT2TEXT_GENERATION = enum.auto()


def import_model(
    model_name: str,
    tag: bentoml.Tag,
    __openllm_framework__: str,
    *model_args: t.Any,
    tokenizer_kwds: dict[str, t.Any],
    **attrs: t.Any,
):
    """Auto detect model type from given model_name and import it to bentoml's model store.

    For all kwargs, it will be parsed into `transformers.AutoConfig.from_pretrained` first,
    returning all of the unused kwargs.
    The unused kwargs then parsed directly into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
    For all tokenizer kwargs, make sure to prefix it with `_tokenizer_` to avoid confusion.

    Note: Currently, there are only two tasks supported: `text-generation` and `text2text-generation`.

    Refer to Transformers documentation for more information about kwargs.

    Args:
        model_name: Model name to be imported. use `openllm models` to see available entries
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
                model_name, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_attrs, **copied_attrs
            ),
        )

    if type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING:
        task_type = "text-generation"
    elif type(config) in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
        task_type = "text2text-generation"
    else:
        raise OpenLLMException(f"Model type {type(config)} is not supported yet.")

    return bentoml.transformers.save_model(
        str(tag),
        getattr(
            transformers, _return_tensors_to_framework_map[__openllm_framework__][TaskType[task_type].value - 1]
        ).from_pretrained(
            model_name, *model_args, config=config, trust_remote_code=trust_remote_code, **hub_attrs, **attrs
        ),
        custom_objects={
            "tokenizer": t.cast(
                "LLMTokenizer",
                transformers.AutoTokenizer.from_pretrained(
                    model_name, config=config, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_kwds
                ),
            )
        },
    )


_required_namespace = {"default_model", "variants"}

_reserved_namespace = _required_namespace | {
    "config_class",
    "model",
    "tokenizer",
    "import_kwargs",
    "requirements",
}


class LLMInterface(ABC):
    """This defines the loose contract for all openllm.LLM implementations."""

    default_model: str
    """Return the default model to use when using 'openllm start <model_name>'.
    This could be one of the keys in 'self.variants' or custom users model."""

    variants: list[str]
    """A list of supported pretrainede models tag for this given runnable.

    For example:
        For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                            "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]
    """

    config_class: type[openllm.LLMConfig]
    """The config class to use for this LLM. If you are creating a custom LLM, you must specify this class."""

    import_kwargs: dict[str, t.Any] | None = None
    """The default import kwargs to used when importing the model. 
    This will be passed into 'openllm.LLM.import_model'."""

    requirements: list[str] | None = None
    """The default PyPI requirements needed to run this given LLM. By default, we will depend on 
    bentoml, torch, transformers."""

    def preprocess_parameters(self, prompt: str, **generation_kwds: t.Any) -> tuple[str, dict[str, t.Any]]:
        """This handler will preprocess given prompt into the correct inputs with all
        filled templates that the model accepts. It will returns a tuple of expected LLM prompt type and and dictionary
        of expected arguments to be passed into model.generate.

        You can customize the generation attributes for generate here. By default, it is a simple echo.

        NOTE: this will be used from the client side.
        """
        return prompt, generation_kwds

    @abstractmethod
    def generate(self, prompt: str, **preprocess_generate_kwds: t.Any) -> t.Any:
        """The main function implementation for generating from given prompt. It takes the outputs
        of preprocess_generate, and then those can be pass to 'model.generate'.
        """
        raise NotImplementedError

    def postprocess_parameters(self, prompt: str, generation_result: t.Any, **attrs: t.Any) -> t.Any:
        """This handler will postprocess generation results from LLM.generate and
        then output nicely formatted results (if the LLM decide to do so.)

        You can customize how the output of the LLM looks with this hook. By default, it is a simple echo.

        NOTE: this will be used from the client side.
        """
        return generation_result

    def generate_iterator(self, prompt: str, **attrs: t.Any) -> t.Iterator[t.Any]:
        """An iterator version of generate function."""
        raise NotImplementedError(
            "Currently generate_iterator requires SSE (Server-side events) support, which is not yet implemented."
        )

    def llm_post_init(self):
        """This function can be implemented if you need to initialized any additional variables that doesn't
        concern OpenLLM internals.
        """
        pass

    def import_model(
        self,
        pretrained: str,
        tag: bentoml.Tag,
        *args: t.Any,
        tokenizer_kwds: dict[str, t.Any],
        **attrs: t.Any,
    ) -> bentoml.Model:
        """This function can be implemented if default import_model doesn't satisfy your needs."""
        raise NotImplementedError


class LLMMetaclass(ABCMeta):
    def __new__(
        mcls, cls_name: str, bases: tuple[type[t.Any], ...], namespace: dict[str, t.Any], **attrs: t.Any
    ) -> type:
        """Metaclass for creating a LLM."""
        if LLMInterface not in bases:  # only actual openllm.LLM should hit this branch.
            if "__annotations__" not in namespace:
                annotations_dict: dict[str, t.Any] = {}
                namespace["__annotations__"] = annotations_dict

            # NOTE: check for required attributes
            for k in _required_namespace:
                if k not in namespace:
                    raise RuntimeError(f"Missing required key '{k}'. Make sure to define it within the LLM subclass.")

            # NOTE: set implementation branch
            prefix_class_name_config = cls_name
            if "__llm_implementation__" in namespace:
                raise RuntimeError(
                    f"""\
                __llm_implementation__ should not be set directly. Instead make sure that your class
                name follows the convention prefix: 
                - For Tensorflow implementation: 'TF{cls_name}'
                - For Flax implementation: 'Flax{cls_name}'
                - For PyTorch implementation: '{cls_name}'"""
                )
            if cls_name.startswith("Flax"):
                implementation = "flax"
                prefix_class_name_config = cls_name[4:]
                namespace["__annotations__"].update({"model": "transformers.FlaxPreTrainedModel"})
            elif cls_name.startswith("TF"):
                implementation = "tf"
                prefix_class_name_config = cls_name[2:]
                namespace["__annotations__"].update({"model": "transformers.TFPreTrainedModel"})
            else:
                implementation = "pt"
                namespace["__annotations__"].update({"model": "transformers.PreTrainedModel"})
            namespace["__llm_implementation__"] = implementation

            # NOTE: setup config class branch
            if "__openllm_internal__" in namespace:
                # NOTE: we will automatically find the subclass for this given config class
                if "config_class" not in namespace:
                    # this branch we will automatically get the class
                    namespace["config_class"] = getattr(openllm, f"{prefix_class_name_config}Config")
                else:
                    logger.debug(f"Using config class {namespace['config_class']} for {cls_name}.")

            config_class: type[openllm.LLMConfig] = namespace["config_class"]
            # Run check for GPU
            config_class.check_if_gpu_is_available(namespace["__llm_implementation__"])

            for key in ("__openllm_start_name__", "__openllm_requires_gpu__"):
                namespace[key] = getattr(config_class, key)

            # NOTE: import_model branch
            if "import_model" not in namespace:
                # using the default import model
                namespace["import_model"] = functools.partial(
                    import_model, __openllm_framework__=namespace["__llm_implementation__"]
                )
            else:
                logger.debug("Using custom 'import_model' for %s", cls_name)

            # NOTE: populate with default cache.
            namespace.update({k: None for k in ("__llm_bentomodel__", "__llm_model__", "__llm_tokenizer__")})

            cls: type[LLM] = super().__new__(t.cast("type[type[LLM]]", mcls), cls_name, bases, namespace, **attrs)
            cls.__openllm_post_init__ = None if cls.llm_post_init is LLMInterface.llm_post_init else cls.llm_post_init

            if getattr(cls, "config_class") is None:
                raise RuntimeError(f"'config_class' must be defined for '{cls.__name__}'")
            return cls
        else:
            # the LLM class itself being created, no need to setup
            return super().__new__(mcls, cls_name, bases, namespace, **attrs)


class LLM(LLMInterface, metaclass=LLMMetaclass):
    if t.TYPE_CHECKING:
        # NOTE: the following will be populated by metaclass
        __llm_bentomodel__: bentoml.Model | None = None
        __llm_model__: LLMModel | None = None
        __llm_tokenizer__: LLMTokenizer | None = None
        __llm_implementation__: t.Literal["pt", "tf", "flax"]
        __llm_kwargs__: dict[str, t.Any]
        __llm_args__: tuple[t.Any, ...]

        __openllm_start_name__: str
        __openllm_requires_gpu__: bool
        __openllm_post_init__: t.Callable[[t.Self], None] | None

        # NOTE: the following will be populated by __init__
        config: openllm.LLMConfig

    def __init__(
        self,
        pretrained: str | None = None,
        llm_config: openllm.LLMConfig | None = None,
        *args: t.Any,
        **attrs: t.Any,
    ):
        """Initialize the LLM with given pretrained model.

        Note:
        - *args to be passed to the model.
        - **attrs will first be parsed to the AutoConfig, then the rest will be parsed to the import_model
        - for tokenizer kwargs, it should be prefixed with _tokenizer_*

        Current drawback with pretrained is that we don't have support loading from custom files yet.
        We actually use the commit_hash to generate the model version, therefore, we can't use custom files.
        Current recommendation is to push the model onto huggingface hub, then use such tag to load with the model.

        If you need to overwrite the default ``import_model``, implement the following in your subclass:

        ```python
        def import_model(
            self,
            pretrained: str,
            tag: bentoml.Tag,
            *args: t.Any,
            tokenizer_kwds: dict[str, t.Any],
            **attrs: t.Any,
        ):
            return bentoml.transformers.save_model(
                str(tag),
                transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained, device_map="auto", torch_dtype=torch.bfloat16, **attrs
                ),
                custom_objects={
                    "tokenizer": transformers.AutoTokenizer.from_pretrained(
                        pretrained, padding_size="left", **tokenizer_kwds
                    )
                },
            )
        ```

        If your import model doesn't require customization, you can simply pass in `import_kwargs`
        at class level that will be then passed into The default `import_model` implementation.
        See ``openllm.DollyV2`` for example.

        ```python
        dolly_v2_runner = openllm.Runner(
            "dolly-v2", _tokenizer_padding_size="left", torch_dtype=torch.bfloat8, device_map="gpu"
        )
        ```

        Note: If you implement your own `import_model`, then `import_kwargs` will be the
        default kwargs for every load. You can still override those via ``openllm.Runner``.

        Note that this tag will be generated based on `self.default_model` or the given `pretrained` kwds.
        passed from the __init__ constructor.

        ``llm_post_init`` can also be implemented if you need to do any
        additional initialization after everything is setup.

        Args:
            pretrained: The pretrained model to use. Defaults to None. It will use 'self.default_model' if None.
            llm_config: The config to use for this LLM. Defaults to None. If not passed, we will use 'self.config_class'
                        to construct default configuration.
            *args: The args to be passed to the model.
            **attrs: The kwargs to be passed to the model.
        """

        if llm_config is not None:
            logger.debug("Using given 'llm_config=(%s)' to initialize LLM", llm_config)
            self.config = llm_config
        else:
            self.config = self.config_class(**attrs)
            assert self.config.__pydantic_extra__ is not None
            # The rests of the kwargs that is not used by the config class should
            # be stored into __pydantic_extra__.
            attrs = copy.deepcopy(self.config.__pydantic_extra__)

        if pretrained is None:
            pretrained = os.environ.get(f"OPENLLM_{self.config.__openllm_model_name__.upper()}_PRETRAINED", None)
            if not pretrained:
                assert self.default_model, "A default model is required for any LLM."
                pretrained = self.default_model

        # NOTE: This is the actual given path or pretrained weight for this LLM.
        self._pretrained = pretrained

        # NOTE: Save the args and kwargs for latter load
        self.__llm_args__ = args
        self.__llm_kwargs__ = attrs

        if self.__openllm_post_init__:
            self.__openllm_post_init__(self)

    def __setattr__(self, attr: str, value: t.Any):
        if attr in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"{attr} should not be set during runtime "
                f"as these value will be reflected during runtime. "
                f"Instead, you can create a custom LLM subclass {self.__class__.__name__}."
            )

        super().__setattr__(attr, value)

    # NOTE: The section below defines a loose contract with langchain's LLM interface.
    @property
    def llm_type(self) -> str:
        assert self.default_model is not None
        return openllm.utils.convert_transformers_model_name(self.default_model)

    @property
    def identifying_params(self) -> dict[str, t.Any]:
        return {"configuration": self.config.model_dump(), "variants": self.variants}

    @property
    def _bentomodel(self) -> bentoml.Model:
        if self.__llm_bentomodel__ is None:
            trust_remote_code = self.__llm_kwargs__.pop("trust_remote_code", self.config.__openllm_trust_remote_code__)
            tag, kwds = openllm.utils.generate_tags(
                self._pretrained,
                prefix=self.__llm_implementation__,
                trust_remote_code=trust_remote_code,
                **self.__llm_kwargs__,
            )

            tokenizer_kwds = {k[len("_tokenizer_") :]: v for k, v in kwds.items() if k.startswith("_tokenizer_")}
            kwds = {k: v for k, v in kwds.items() if not k.startswith("_tokenizer_")}

            if self.import_kwargs:
                tokenizer_kwds = {
                    **{
                        k[len("_tokenizer_") :]: v for k, v in self.import_kwargs.items() if k.startswith("_tokenizer_")
                    },
                    **tokenizer_kwds,
                }
                kwds = {
                    **{k: v for k, v in self.import_kwargs.items() if not k.startswith("_tokenizer_")},
                    **kwds,
                }

            try:
                self.__llm_bentomodel__ = bentoml.transformers.get(tag)
            except bentoml.exceptions.BentoMLException:
                logger.info(
                    "'%s' with tag (%s) not found, importing from HuggingFace Hub.", self.__class__.__name__, tag
                )
                self.__llm_bentomodel__ = self.import_model(
                    self._pretrained,
                    tag,
                    *self.__llm_args__,
                    tokenizer_kwds=tokenizer_kwds,
                    trust_remote_code=trust_remote_code,
                    **kwds,
                )
        return self.__llm_bentomodel__

    @property
    def tag(self) -> bentoml.Tag:
        return self._bentomodel.tag

    @property
    def model(self) -> LLMModel:
        """The model to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        if self.import_kwargs:
            kwds = {
                **{k: v for k, v in self.import_kwargs.items() if not k.startswith("_tokenizer_")},
                **self.__llm_kwargs__,
            }
        else:
            kwds = self.__llm_kwargs__

        if self.__llm_model__ is None:
            # Hmm, bentoml.transformers.load_model doesn't yet support args.
            self.__llm_model__ = self._bentomodel.load_model(*self.__llm_args__, **kwds)
        return self.__llm_model__

    @property
    def tokenizer(self) -> LLMTokenizer:
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

    def to_runner(
        self,
        name: str | None = None,
        models: list[bentoml.Model] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, ModelSignatureDict | ModelSignature] | None = None,
        embedded: bool = False,
        scheduling_strategy: type[Strategy] | None = None,
    ) -> bentoml.Runner:
        """Convert this LLM into a Runner.

        Args:
            name: The name of the runner to generate. Optional as this will be generated based on the model_name.
            models: Any additional ``bentoml.Model`` to be included in this given models.
                    By default, this will be determined from the model_name.
            max_batch_size: The maximum batch size for the runner.
            max_latency_ms: The maximum latency for the runner.
            method_configs: The method configs for the runner.
            strategy: The strategy to use for this runner.
            embedded: Whether to run this runner in embedded mode.
            scheduling_strategy: Whether to create a custom scheduling strategy for this Runner.
        """

        if name is None:
            name = f"llm-{self.config.__openllm_start_name__}-runner"
        models = models if models is not None else []

        # NOTE: The side effect of this is that will load the imported model during runner creation.
        models.append(self._bentomodel)

        if scheduling_strategy is None:
            from bentoml._internal.runner.strategy import DefaultStrategy

            scheduling_strategy = DefaultStrategy

        generate_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=False))
        generate_iterator_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=True))
        if method_configs is None:
            method_configs = {"generate": generate_sig, "generate_iterator": generate_iterator_sig}
        else:
            generate_sig = ModelSignature.convert_signatures_dict(method_configs).get("generate", generate_sig)
            generate_iterator_sig = ModelSignature.convert_signatures_dict(method_configs).get(
                "generate_iterator", generate_iterator_sig
            )

        class _Runnable(bentoml.Runnable):
            SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
            SUPPORTS_CPU_MULTI_THREADING = True

            llm_type: str
            identifying_params: dict[str, t.Any]

            def __init_subclass__(cls, **attrs: t.Any):
                cls.llm_type = self.llm_type
                cls.identifying_params = self.identifying_params
                super().__init_subclass__(**attrs)

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

        if self.__openllm_requires_gpu__:
            _supported_resources = ("nvidia.com/gpu",)
        else:
            _supported_resources = ("nvidia.com/gpu", "cpu")

        return bentoml.Runner(
            type(
                inflection.camelize(self.config.__openllm_model_name__) + "Runnable",
                (_Runnable,),
                {"SUPPORTED_RESOURCES": _supported_resources},
            ),
            name=name,
            models=models,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=bentoml_cattr.unstructure(method_configs),
            embedded=embedded,
            scheduling_strategy=scheduling_strategy,
        )


def Runner(start_name: str, **attrs: t.Any) -> bentoml.Runner:
    """Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'

    Args:
        start_name: Supported model name from 'openllm models'
        init_local: Whether to init_local this given Runner. This is useful during development. (Default to False)
        **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs
                behaviour
    """
    init_local = attrs.pop("init_local", False)
    envvar = openllm.utils.get_framework_env(start_name)
    if envvar == "flax":
        runner = openllm.AutoFlaxLLM.create_runner(start_name, **attrs)
    elif envvar == "tf":
        runner = openllm.AutoTFLLM.create_runner(start_name, **attrs)
    else:
        runner = openllm.AutoLLM.create_runner(start_name, **attrs)

    if init_local:
        runner.init_local()

    return runner
