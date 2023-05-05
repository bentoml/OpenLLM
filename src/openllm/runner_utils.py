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
OpenLLM Runnable interface. This define a loose contract for LLMRunnable, which can then be implemented per any model.

LLMRunnable also follow a loose API definition from LangChain's LLM, therefore it can also be used in conjunction with LangChain.
"""
from __future__ import annotations

import logging
import typing as t
from abc import ABC, abstractmethod

import attr
import bentoml

import openllm

from .configuration_utils import LLMConfig, ModelSignature
from .utils import _object_setattr

if t.TYPE_CHECKING:
    import transformers

    from openllm.types import (InferenceConfig, LLMModuleType,
                               ModelSignatureDict, ModelSignaturesType,
                               TokenizerRunner)

else:
    InferenceConfig = ModelSignatureDict = dict

    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)


def assign_start_model_name(start_model_name: str) -> t.Callable[..., t.Any]:
    def wrapper(fn: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        _object_setattr(fn, "start_model_name", start_model_name)
        return fn

    return wrapper


def generate_tokenizer_runner(
    tokenizer: transformers.PreTrainedTokenizer
    | transformers.PreTrainedTokenizerBase
    | transformers.PreTrainedTokenizerFast,
    embedded: bool = False,
) -> TokenizerRunner:
    """Generate a runner from any given transformers.AutoTokenizer.

    Args:
        tokenizer: The tokenizer to generate the runner from.
    """

    # NOTE: I'm going to maintain this function from bentoml side, so internal imports here.
    from bentoml._internal.frameworks.transformers import \
        make_default_signatures

    signatures: ModelSignaturesType = make_default_signatures(tokenizer)

    def __init_runnable__(self: bentoml.Runnable):
        # keep a reference object to the tokenizer
        self._tokenizer = tokenizer

        self._predict_fns = {}
        for method_name in signatures:
            self._predict_fns[method_name] = getattr(tokenizer, method_name)

    def add_method(cls: type[bentoml.Runnable], method_name: str, options: ModelSignature):
        def fn(self: bentoml.Runnable, *args: t.Any, **kwargs: t.Any) -> t.Any:
            try:
                return self._predict_fns[method_name](*args, **kwargs)
            except KeyError:
                raise bentoml.exceptions.BadInput(f"Method {method_name} is not supported by this tokenizer.")

        cls.add_method(
            fn,
            method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    RunnableCls: type[bentoml.Runnable] = type(
        f"{tokenizer.__class__.__qualname__}Runnable",
        (bentoml.Runnable,),
        {
            "SUPPORTED_RESOURCES": ("cpu",),
            "SUPPORTS_CPU_MULTI_THREADING": True,
            "__init__": __init_runnable__,
        },
    )

    for method_name, options in signatures.items():
        add_method(RunnableCls, method_name, options)

    return t.cast(
        "TokenizerRunner",
        bentoml.Runner(RunnableCls, name=f"{tokenizer.__class__.__qualname__.lower()}-runner", embedded=embedded),
    )


class BaseLLMRunnable(bentoml.Runnable, ABC):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    # NOTE: A loose contract for LLMRunnable implementations.
    ATTACH_TOKENIZER: bool = True
    """This boolean determine whether to attach a tokenizer runner to this runnable or not. 
    By default, it is True for _generate."""

    start_model_name: str
    """The default model to use when using ``openllm start <model_name>``."""

    default_model: str | None = None
    """Return the default model to use when using ``openllm start <model_name>``.
    This could be one of the keys in self.variants or custom users model."""

    inference_config: InferenceConfig = InferenceConfig(generate=ModelSignatureDict(batchable=False))
    """The inference config for the two endpoints of this model.
    This is the equivalent of args that is passed into ``bentoml.Runnable.method``.
    """

    config_class: type[LLMConfig] = LLMConfig
    """The config class for any given LLMRunnable implementation."""

    variants: list[str]
    """A list of supported pretrainede models tag for this given runnable.

    For example:
        For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                            "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]
    """

    @abstractmethod
    def _generate(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """This method should be implemented to provide a generation interface for the given model."""
        raise NotImplementedError


# TODO: Add support for model validation
class LLMRunnable(BaseLLMRunnable):
    # The section below defines a loose contract with langchain's LLM interface.
    @property
    def _llm_type(self) -> str:
        assert self.default_model is not None
        return openllm.utils.convert_transformers_model_name(self.default_model)

    @property
    def _identifying_params(self) -> dict[str, t.Any]:
        return {
            "configuration": self._llm_config.dict(),
            "inference_config": self.inference_config,
            "pretrained": self.pretrained_models(),
        }

    # XXX: INTERNAL
    _module: LLMModuleType
    _model: t.Any | None = None
    _tokenizer: t.Any | None = None

    def __setattr__(self, attr_name: str, value: t.Any) -> None:
        if attr_name in ("ATTACH_TOKENIZER",):
            raise openllm.exceptions.OpenLLMException(
                f"{attr_name} should not be set at runtime, as it determines how the runner is created. \
                Please create a separate Runnable that extends '{self}' instead."
            )
        super().__setattr__(attr_name, value)

    def __init_subclass__(cls, *, start_model_name: str, module: str | None = None):
        cls.start_model_name = start_model_name

        if module is None:
            module = cls.__module__
        cls._module = t.cast("LLMModuleType", openllm.utils.LazyLoader(module, globals(), module))

    def __init__(
        self,
        /,
        *,
        model_name: str | None = None,
        _bentomodel: bentoml.Model | None = None,
        _llm_config: LLMConfig | None = None,
        _internal: bool = False,
        _dummy: bool = False,
        **kwargs: t.Any,
    ):
        self._dummy = _dummy
        if _dummy:
            if not _internal:
                raise openllm.exceptions.ForbiddenAttributeError(
                    "'_dummy' should only be used for internal, not public facing."
                )
            return

        if _bentomodel is not None:
            if not _internal:
                raise openllm.exceptions.ForbiddenAttributeError(
                    "'_bentomodel' should only be used for internal, not public facing."
                )
            self._bentomodel = _bentomodel
        else:
            import_model_kwargs = kwargs.pop("import_model_kwargs", {})
            import_tokenizer_kwargs = kwargs.pop("import_tokenizer_kwargs", {})
            import_config_kwargs = kwargs.pop("import_config_kwargs", {})

            assert self.default_model is not None
            model_name = self.default_model if model_name is None else model_name

            self._bentomodel = self._module.import_model(
                model_name,
                model_kwargs=import_model_kwargs,
                tokenizer_kwargs=import_tokenizer_kwargs,
                config_kwargs=import_config_kwargs,
            )

        if _llm_config is not None:
            if not _internal:
                raise openllm.exceptions.ForbiddenAttributeError(
                    "'_llm_config' should only be used for internal, not public facing."
                )
            self._llm_config = _llm_config
        else:
            assert self.config_class is not None, "'config_class' is required."
            self._llm_config = self.config_class(**kwargs)

    def __getattribute__(self, item: t.Any):
        is_dummy = object.__getattribute__(self, "_dummy")
        if is_dummy and item in ("model", "tokenizer", "create_runner", "bentoml_runnable_methods__", "add_method"):
            logger.warning(f"Accessing '{item}' on dummy object. Returning None.")
            return
        return super().__getattribute__(item)

    @classmethod
    def dummy_object(cls) -> LLMRunnable:
        return cls(_dummy=True, _internal=True)

    @property
    def model(self) -> t.Any:
        # NOTE: should we have support for nested runner here?
        if self._model is None:
            self._model = self._bentomodel.load_model()
        return self._model

    @property
    def tokenizer(self) -> t.Any:
        # This is the runner generated from the bento model. This can
        # then be used for implementation of _generate.
        if self._tokenizer is None:
            if not self.ATTACH_TOKENIZER:
                raise openllm.exceptions.OpenLLMException(
                    "Tokenizer runner is not attached. Please set ATTACH_TOKENIZER=True when creating this runnable."
                )
            self._tokenizer = self._bentomodel.custom_objects["tokenizer"]
        return self._tokenizer

    @classmethod
    def create_runner(
        cls,
        pretrained_or_path: str | None = None,
        config: LLMConfig | None = None,
        runner_name: str | None = None,
        models: list[bentoml.Model] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: ModelSignaturesType | None = None,
        embedded: bool = False,
        import_model_kwargs: dict[str, t.Any] | None = None,
        import_tokenizer_kwargs: dict[str, t.Any] | None = None,
        import_config_kwargs: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> LLMRunner:
        """Convert this LLMRunnable to a bentoml.Runner.

        Args:
            model_name: The name of the model to generate the runner from.
            runner_name: The name of the runner to generate. Optional as this will be generated based on the model_name.
            models: Any additional ``bentoml.Model`` to be included in this given models. By default, this will be determined from the model_name.
            max_batch_size: The maximum batch size for the runner.
            max_latency_ms: The maximum latency for the runner.
            method_configs: The method configs for the runner.
            embedded: Whether to run this runner in embedded mode.
            import_model_kwargs: To pass to model_kwargs in ``import_model``.
            import_tokenizer_kwargs: To pass to tokenizer_kwargs in ``import_model``.
            import_config_kwargs: To pass to config_kwargs in ``import_model``.
            The rest of the kwargs will then be passed into ``self.config_class`` (which is of type ``LLMConfig``).

        Returns:
            A bentoml.Runner instance.
        """
        if pretrained_or_path is None:
            if cls.default_model is None:
                raise openllm.exceptions.OpenLLMException(
                    "A default model is required for any LLMRunnable. Make sure to specify a default_model or pass in a model_name."
                )
            pretrained_or_path = cls.default_model
            if pretrained_or_path not in cls.variants:
                logger.debug("Creating runner for custom model '%s'", cls.default_model)

        models = models or []
        bentomodel = cls._module.import_model(
            pretrained_or_path,
            model_kwargs=import_model_kwargs,
            tokenizer_kwargs=import_tokenizer_kwargs,
            config_kwargs=import_config_kwargs,
        )
        models.append(bentomodel)

        if runner_name is None:
            runner_name = f"llm-{cls.start_model_name}-runner"

        _bento_runnable_methods_map = {"generate": cls._generate}

        for method_name, method_config in cls.inference_config.items():
            signature = ModelSignature.from_dict(t.cast(ModelSignatureDict, method_config))
            cls.add_method(
                _bento_runnable_methods_map[method_name],
                method_name,
                batchable=signature.batchable,
                batch_dim=signature.batch_dim,
                input_spec=signature.input_spec,
                output_spec=signature.output_spec,
            )

        # The rest of the kwargs can be then pass to LLMConfig
        if config is not None:
            llm_config = config.with_options(**kwargs)
        else:
            llm_config = cls.config_class(**kwargs)

        return LLMRunner(
            cls,
            llm_config=llm_config,
            runnable_init_params={
                "model_name": pretrained_or_path,
                "_bentomodel": bentomodel,
                "_llm_config": llm_config,
                "_internal": True,
            },
            name=runner_name,
            models=models,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=method_configs,
            embedded=embedded,
        )


class LLMRunner(bentoml.Runner):
    llm_config: LLMConfig = attr.field(factory=lambda: LLMConfig())

    def __init__(
        self,
        runnable_class: type[LLMRunnable],
        llm_config: LLMConfig,
        **kwargs: t.Any,
    ):
        super().__init__(runnable_class, **kwargs)
        # A hack around frozen attributes.
        _object_setattr(self, "llm_config", llm_config)
