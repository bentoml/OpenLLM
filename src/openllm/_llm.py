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
import logging
import os
import types
import typing as t
from abc import ABC, abstractmethod

import bentoml
import inflection

import openllm

from ._configuration import ModelSignature
from .exceptions import ForbiddenAttributeError, OpenLLMException
from .utils import cattr

if t.TYPE_CHECKING:
    import transformers
    from bentoml._internal.runner.strategy import Strategy

    from .types import LLMModel, LLMTokenizer, ModelSignatureDict
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


# NOTE: Currently, all LLMs are either text-generation or text2text-generation
# hence, the two dicts to check are
# transformers.MODEL_FOR_CAUSAL_LM_MAPPING & transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
def import_model(model_name: str, tag: bentoml.Tag, *model_args: t.Any, **kwds: t.Any):
    """Auto detect model type from given model_name and import it to bentoml's model store."""
    _framework_impl = kwds.pop("_for_framework", "pt")
    config: transformers.PretrainedConfig = kwds.pop("config", None)
    trust_remote_code = kwds.pop("trust_remote_code", False)

    tokenizer_kwds = {k[len("_tokenizer_") :]: v for k, v in kwds.items() if k.startswith("_tokenizer_")}

    kwds = {k: v for k, v in kwds.items() if not k.startswith("_tokenizer_")}
    # this logic below is synonymous to handling `from_pretrained` kwds.
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
    hub_kwds = {k: kwds.pop(k) for k in hub_kwds_names if k in kwds}
    if not isinstance(config, transformers.PretrainedConfig):
        copied_kwds = copy.deepcopy(kwds)
        if copied_kwds.get("torch_dtype", None) == "auto":
            copied_kwds.pop("torch_dtype")
        config, kwds = t.cast(
            "tuple[transformers.PretrainedConfig, dict[str, t.Any]]",
            transformers.AutoConfig.from_pretrained(
                model_name, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_kwds, **copied_kwds
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
            transformers, _return_tensors_to_framework_map[_framework_impl][TaskType[task_type].value - 1]
        ).from_pretrained(
            model_name, *model_args, config=config, trust_remote_code=trust_remote_code, **hub_kwds, **kwds
        ),
        custom_objects={
            "tokenizer": t.cast(
                "LLMTokenizer",
                transformers.AutoTokenizer.from_pretrained(
                    model_name, config=config, trust_remote_code=trust_remote_code, **hub_kwds, **tokenizer_kwds
                ),
            )
        },
    )


_reserved_namespace = {"default_model", "variants", "config_class", "model", "tokenizer"}


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

    @abstractmethod
    def generate(self, prompt: str, **kwargs: t.Any) -> t.Any:
        """The main function implementation for generating from given prompt."""
        raise NotImplementedError

    def generate_iterator(self, prompt: str, **kwargs: t.Any) -> t.Iterator[t.Any]:
        """An iterator version of generate function."""
        raise NotImplementedError


if t.TYPE_CHECKING:

    class LLMRunnable(bentoml.Runnable):
        @abstractmethod
        def generate(self, prompt: str, **kwargs: t.Any) -> t.Any:
            ...


class LLM(LLMInterface):
    _implementation: t.Literal["pt", "tf", "flax"]

    __bentomodel__: bentoml.Model | None = None
    __llm_model__: LLMModel | None = None
    __llm_tokenizer__: LLMTokenizer | None = None
    __openllm_start_name__: str = ""

    if t.TYPE_CHECKING:

        def import_model(self, pretrained: str, tag: bentoml.Tag, *model_args: t.Any, **kwds: t.Any) -> bentoml.Model:
            ...

    def __init_subclass__(cls, *, implementation: t.Literal["pt", "tf", "flax"] = "pt", _internal: bool = False):
        cls._implementation = implementation
        if not _internal and getattr(cls, "config_class", None) is None:
            raise RuntimeError("'config_class' must be defined for LLM subclasses.")
        else:
            if getattr(cls, "config_class", None) is None:
                if implementation == "tf":
                    cls.config_class = getattr(openllm, f"{cls.__name__[2:]}Config")
                elif implementation == "flax":
                    cls.config_class = getattr(openllm, f"{cls.__name__[len('flax'):]}Config")
                else:
                    cls.config_class = getattr(openllm, f"{cls.__name__}Config")
            else:
                logger.debug(f"Using config class {cls.config_class} for {cls.__name__}.")

        cls.__openllm_start_name__ = cls.config_class.__openllm_start_name__

    def __setattr__(self, attr: str, value: t.Any):
        if attr in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"{attr} should not be set during runtime "
                f"as these value will be reflected during runtime. "
                f"Instead, you can create a custom LLM subclass {self.__class__.__name__}."
            )

        super().__setattr__(attr, value)

    # The section below defines a loose contract with langchain's LLM interface.
    @property
    def llm_type(self) -> str:
        assert self.default_model is not None
        return openllm.utils.convert_transformers_model_name(self.default_model)

    @property
    def identifying_params(self) -> dict[str, t.Any]:
        return {"configuration": self.config.model_dump(), "variants": self.variants}

    def __init__(
        self, pretrained: str | None = None, llm_config: openllm.LLMConfig | None = None, *args: t.Any, **kwargs: t.Any
    ):
        """Initialize the LLM with given pretrained model.

        Note:
        - *args to be passed to the model.
        - **kwargs will first be parsed to the AutoConfig, then the rest will be parsed to the import_model
        - for tokenizer kwargs, it should be prefixed with _tokenizer_*

        Current drawback with pretrained is that we don't have support loading from custom files yet.
        We actually use the commit_hash to generate the model version, therefore, we can't use custom files.
        Current recommendation is to push the model onto huggingface hub, then use such tag to load with the model.

        If you need to overwrite the default ``import_model``, implement the following in your subclass:

        ```python
        def import_model(self, pretrained: str, tag: bentoml.Tag, *args: t.Any, **kwargs: t.Any) -> bentoml.Model:
            return bentoml.transformers.save_model(str(tag), ...)
        ```

        Note: See ``openllm.DollyV2`` for example

        Note that this tag will be generated based on `self.default_model` or the given `pretrained` kwds.
        passed from the __init__ constructor.

        Args:
            pretrained: The pretrained model to use. Defaults to None. It will use self.default_model if None.
            llm_config: The config to use for this LLM. Defaults to None. It will use self.config_class to construct default configuration.
            *args: The args to be passed to the model.
            **kwargs: The kwargs to be passed to the model.
        """

        if llm_config is not None:
            self.config = llm_config
        else:
            self.config = self.config_class(**kwargs)
            assert self.config.__pydantic_extra__ is not None
            # The rests of the kwargs that is not used by the config class should
            # be stored into __pydantic_extra__.
            kwargs = copy.deepcopy(self.config.__pydantic_extra__)

        if pretrained is None:
            pretrained = os.environ.get(f"OPENLLM_{self.config.__openllm_model_name__.upper()}_PRETRAINED", None)
            if not pretrained:
                assert self.default_model, "A default model is required for any LLM."
                pretrained = self.default_model

        self._pretrained = pretrained
        # NOTE: Save the args and kwargs for latter load
        self._args = args
        self._kwargs = kwargs

    @property
    def _bentomodel(self) -> bentoml.Model:
        if self.__bentomodel__ is None:
            tag, kwargs = openllm.utils.generate_tags(self._pretrained, prefix=self._implementation, **self._kwargs)
            try:
                self.__bentomodel__ = bentoml.transformers.get(tag)
            except bentoml.exceptions.BentoMLException:
                logger.info(
                    "'%s' with tag (%s) not found, importing from HuggingFace Hub.", self.__class__.__name__, tag
                )
                if hasattr(self, "import_model"):
                    logger.debug("Using custom 'import_model' defined in subclass.")
                    self.__bentomodel__ = self.import_model(self._pretrained, tag, *self._args, **kwargs)
                else:
                    self._kwargs["_for_framework"] = self._implementation
                    # In this branch, we just use the default implementation.
                    self.__bentomodel__ = import_model(self._pretrained, tag, *self._args, **kwargs)
        return self.__bentomodel__

    @property
    def tag(self) -> bentoml.Tag:
        return self._bentomodel.tag

    @property
    def model(self) -> LLMModel:
        """The model to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        if self.__llm_model__ is None:
            # Hmm, bentoml.transformers.load_model doesn't yet support args.
            self.__llm_model__ = self._bentomodel.load_model(*self._args, **self._kwargs)
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
                    the tokenizer within the model via 'custom_objects'."
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
        **kwargs: t.Any,
    ) -> bentoml.Runner:
        """Convert this LLM into a Runner.

        Args:
            name: The name of the runner to generate. Optional as this will be generated based on the model_name.
            models: Any additional ``bentoml.Model`` to be included in this given models. By default, this will be determined from the model_name.
            max_batch_size: The maximum batch size for the runner.
            max_latency_ms: The maximum latency for the runner.
            method_configs: The method configs for the runner.
            strategy: The strategy to use for this runner.
            embedded: Whether to run this runner in embedded mode.
            kwargs: Any additional kwargs will be then passed to LLM. Consult LLM.__init__() for more information.
        """

        if name is None:
            name = f"llm-{self.config.__openllm_start_name__}-runner"
        models = models if models is not None else []

        # NOTE: The side effect of this is that i will load the imported model during runner creation.
        models.append(self._bentomodel)

        if scheduling_strategy is None:
            from bentoml._internal.runner.strategy import DefaultStrategy

            scheduling_strategy = DefaultStrategy

        signature = ModelSignature.from_dict(ModelSignatureDict(batchable=False))
        if method_configs is None:
            method_configs = {"generate": signature}
        else:
            signature = ModelSignature.convert_signatures_dict(method_configs).get("generate", signature)

        class _Runnable(bentoml.Runnable):
            SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
            SUPPORTS_CPU_MULTI_THREADING = True

            @bentoml.Runnable.method(
                batchable=signature.batchable,
                batch_dim=signature.batch_dim,
                input_spec=signature.input_spec,
                output_spec=signature.output_spec,
            )
            def generate(__self, prompt: str, **kwds: t.Any) -> list[str]:
                return self.generate(prompt, **kwds)

        return bentoml.Runner(
            t.cast(
                "type[LLMRunnable]",
                types.new_class(inflection.camelize(self.config.__openllm_model_name__) + "Runnable", (_Runnable,)),
            ),
            runnable_init_params=kwargs,
            name=name,
            models=models,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=cattr.unstructure(method_configs),
            embedded=embedded,
            scheduling_strategy=scheduling_strategy,
        )


def Runner(start_name: str, **kwds: t.Any):
    envvar = openllm.utils.get_framework_env(start_name)
    if envvar == "flax":
        return openllm.AutoFlaxLLM.create_runner(start_name, **kwds)
    elif envvar == "tf":
        return openllm.AutoTFLLM.create_runner(start_name, **kwds)
    else:
        return openllm.AutoLLM.create_runner(start_name, **kwds)
