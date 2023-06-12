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
import sys
import types
import typing as t
from abc import ABC, ABCMeta, abstractmethod

import bentoml
import inflection
import orjson
from bentoml._internal.models.model import ModelSignature
from bentoml._internal.types import ModelSignatureDict

import openllm

from .exceptions import ForbiddenAttributeError, OpenLLMException
from .models.auto import AutoConfig
from .utils import ENV_VARS_TRUE_VALUES, LazyLoader, bentoml_cattr

if t.TYPE_CHECKING:
    import torch
    import transformers
    from bentoml._internal.runner.strategy import Strategy

    class LLMRunner(bentoml.Runner):
        llm: openllm.LLM[t.Any, t.Any]
        config: openllm.LLMConfig
        llm_type: str
        identifying_params: dict[str, t.Any]

        def __call__(self, *args: t.Any, **attrs: t.Any) -> t.Any:
            ...

else:
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)

# NOTE: `1-2` -> text-generation and text2text-generation
FRAMEWORK_TO_AUTOCLASS_MAPPING = {
    "pt": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    "tf": ("TFAutoModelForCausalLM", "TFAutoModelForSeq2SeqLM"),
    "flax": ("FlaxAutoModelForCausalLM", "FlaxAutoModelForSeq2SeqLM"),
}


def convert_transformers_model_name(name: str) -> str:
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

    if type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING:
        idx = 0
    elif type(config) in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
        idx = 1
    else:
        raise OpenLLMException(f"Model type {type(config)} is not supported yet.")

    try:
        return bentoml.transformers.save_model(
            tag,
            getattr(
                transformers,
                FRAMEWORK_TO_AUTOCLASS_MAPPING[_model_framework][idx],
            ).from_pretrained(
                model_id,
                *model_args,
                config=config,
                trust_remote_code=trust_remote_code,
                **hub_attrs,
                **attrs,
            ),
            custom_objects={
                "tokenizer": transformers.AutoTokenizer.from_pretrained(
                    model_id,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    **hub_attrs,
                    **tokenizer_kwds,
                )
            },
        )
    finally:
        import gc

        gc.collect()

        # NOTE: We need to free up the cache after importing the model
        # in the case where users first run openllm start without the model
        # available locally.
        if openllm.utils.is_torch_available() and torch.cuda.is_available():
            torch.cuda.empty_cache()


_reserved_namespace = {"config_class", "model", "tokenizer", "import_kwargs"}


class LLMInterface(ABC):
    """This defines the loose contract for all openllm.LLM implementations."""

    config_class: type[openllm.LLMConfig]
    """The config class to use for this LLM. If you are creating a custom LLM, you must specify this class."""

    @property
    def import_kwargs(self) -> dict[str, t.Any] | None:
        """The default import kwargs to used when importing the model.
        This will be passed into 'openllm.LLM.import_model'."""

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


def _default_post_init(self: LLM[t.Any, t.Any]):
    # load_in_mha: Whether to apply BetterTransformer (or Torch MultiHeadAttention) during inference load.
    #              See https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/
    #              for more information.
    # NOTE: set a default variable to transform to BetterTransformer by default for inference
    self.load_in_mha = (
        os.environ.get(self.config_class.__openllm_env__.bettertransformer, str(False)).upper() in ENV_VARS_TRUE_VALUES
    )
    if self.config_class.__openllm_requires_gpu__:
        # For all models that requires GPU, no need to offload it to BetterTransformer
        # use bitsandbytes instead

        self.load_in_mha = False


_M = t.TypeVar("_M")
_T = t.TypeVar("_T")


class LLMMetaclass(ABCMeta):
    def __new__(
        mcls, cls_name: str, bases: tuple[type[t.Any], ...], namespace: dict[str, t.Any], **attrs: t.Any
    ) -> type:
        """Metaclass for creating a LLM."""
        if LLMInterface not in bases:  # only actual openllm.LLM should hit this branch.
            annotations_dict: dict[str, t.Any] = {}

            # NOTE: set implementation branch
            prefix_class_name_config = cls_name
            if cls_name.startswith("Flax"):
                implementation = "flax"
                prefix_class_name_config = cls_name[4:]
                annotations_dict.update(
                    {
                        "__llm_model__": "transformers.FlaxPreTrainedModel",
                        "model": "transformers.FlaxPreTrainedModel",
                    }
                )
            elif cls_name.startswith("TF"):
                implementation = "tf"
                prefix_class_name_config = cls_name[2:]
                annotations_dict.update(
                    {
                        "__llm_model__": "transformers.TFPreTrainedModel",
                        "model": "transformers.TFPreTrainedModel",
                    }
                )
            else:
                implementation = "pt"
                annotations_dict.update(
                    {
                        "__llm_model__": "transformers.PreTrainedModel",
                        "model": "transformers.PreTrainedModel",
                    }
                )
            namespace["__llm_implementation__"] = implementation

            config_class = AutoConfig.infer_class_from_name(prefix_class_name_config)
            annotations_dict["config_class"] = f"type[openllm.{config_class.__qualname__}]"
            annotations_dict["config"] = f"openllm.{config_class.__qualname__}"

            # NOTE: setup config class branch
            if "__openllm_internal__" in namespace:
                # NOTE: we will automatically find the subclass for this given config class
                if "config_class" not in namespace:
                    # this branch we will automatically get the class
                    namespace["config_class"] = config_class
                else:
                    logger.debug(f"Using config class {namespace['config_class']} for {cls_name}.")
            # NOTE: check for required attributes
            else:
                if "config_class" not in namespace:
                    raise RuntimeError(
                        "Missing required key 'config_class'. Make sure to define it within the LLM subclass."
                    )

            # NOTE: the llm_post_init branch
            if "llm_post_init" in namespace:
                original_llm_post_init = namespace["llm_post_init"]

                def wrapped_llm_post_init(self: LLM[t.Any, t.Any]) -> None:
                    """We need to both initialize private attributes and call the user-defined model_post_init
                    method.
                    """
                    _default_post_init(self)
                    original_llm_post_init(self)

                namespace["llm_post_init"] = wrapped_llm_post_init
            else:
                namespace["llm_post_init"] = _default_post_init

            # NOTE: import_model branch
            if "import_model" not in namespace:
                # using the default import model
                namespace["import_model"] = functools.partial(import_model, _model_framework=implementation)
            else:
                logger.debug("Using custom 'import_model' for %s", cls_name)

            namespace.update({k: None for k in {"__llm_model__", "__llm_tokenizer__", "__llm_bentomodel__"}})
            tokenizer_type = "typing.Optional[transformers.PreTrainedTokenizerFast]"
            annotations_dict.update(
                {
                    "load_in_mha": "bool",
                    "tokenizer": tokenizer_type,
                    "__llm_tokenizer__": tokenizer_type,
                    "__llm_bentomodel__": "typing.Optional[bentoml.Model]",
                }
            )
            # NOTE: transformers to ForwardRef
            for key, value in annotations_dict.items():
                if (3, 10) > sys.version_info >= (3, 9, 8) or sys.version_info >= (3, 10, 1):
                    annotations_dict[key] = t.ForwardRef(value, is_argument=False, is_class=True)
                else:
                    annotations_dict[key] = t.ForwardRef(value, is_argument=False)

            namespace["__annotations__"] = annotations_dict

            cls: type[LLM[t.Any, t.Any]] = super().__new__(
                t.cast("type[type[LLM[t.Any, t.Any]]]", mcls), cls_name, bases, namespace, **attrs
            )

            cls.__openllm_post_init__ = None if cls.llm_post_init is LLMInterface.llm_post_init else cls.llm_post_init
            cls.__openllm_custom_load__ = None if cls.load_model is LLMInterface.load_model else cls.load_model
            return cls
        else:
            # the LLM class itself being created, no need to setup
            return super().__new__(mcls, cls_name, bases, namespace, **attrs)


class LLM(LLMInterface, t.Generic[_M, _T], metaclass=LLMMetaclass):
    if t.TYPE_CHECKING:
        # NOTE: the following will be populated by metaclass
        __llm_implementation__: t.Literal["pt", "tf", "flax"]

        __openllm_post_init__: t.Callable[[t.Self], None] | None
        __openllm_custom_load__: t.Callable[[t.Self, t.Any, t.Any], None] | None

        load_in_mha: bool
        _llm_attrs: dict[str, t.Any]
        _llm_args: tuple[t.Any, ...]

    # NOTE: the following is the similar interface to HuggingFace pretrained protocol.
    @classmethod
    def from_pretrained(
        cls, model_id: str | None = None, llm_config: openllm.LLMConfig | None = None, *args: t.Any, **attrs: t.Any
    ) -> LLM[_M, _T]:
        return cls(model_id=model_id, llm_config=llm_config, *args, **attrs)

    def __init__(
        self,
        model_id: str | None = None,
        llm_config: openllm.LLMConfig | None = None,
        *args: t.Any,
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
            *args: The args to be passed to the model.
            **attrs: The kwargs to be passed to the model.

        The following are optional:
            openllm_model_version: version for this `model_id`. By default, users can ignore this if using pretrained
                                   weights as OpenLLM will use the commit_hash of given model_id.
                                   However, if `model_id` is a path, this argument is recomended to include.
        """

        load_in_mha = attrs.pop("load_in_mha", False)

        if llm_config is not None:
            logger.debug("Using given 'llm_config=(%s)' to initialize LLM", llm_config)
            self.config = llm_config
        else:
            self.config = self.config_class(**attrs)
            # The rests of the kwargs that is not used by the config class should be stored into __openllm_extras__.
            attrs = self.config.__openllm_extras__

        if model_id is None:
            model_id = os.environ.get(self.config.__openllm_env__.model_id, self.config.__openllm_default_id__)
            assert model_id is not None

        # NOTE: This is the actual given path or pretrained weight for this LLM.
        self._model_id = model_id

        # NOTE: Save the args and kwargs for latter load
        self._llm_args = args
        self._llm_attrs = attrs

        if self.__openllm_post_init__:
            self.llm_post_init()

        # finally: we allow users to overwrite the load_in_mha defined by the LLM subclass.
        if load_in_mha:
            self.load_in_mha = load_in_mha

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
        return convert_transformers_model_name(self._model_id)

    @property
    def identifying_params(self) -> dict[str, t.Any]:
        return {
            "configuration": self.config.model_dump_json().decode(),
            "model_ids": orjson.dumps(self.config.__openllm_model_ids__).decode(),
        }

    @t.overload
    def make_tag(
        self,
        model_id: str | None = None,
        return_unused_kwargs: t.Literal[False] = ...,
        trust_remote_code: bool = ...,
        **attrs: t.Any,
    ) -> bentoml.Tag:
        ...

    @t.overload
    def make_tag(
        self,
        model_id: str | None = None,
        return_unused_kwargs: t.Literal[True] = ...,
        trust_remote_code: bool = ...,
        **attrs: t.Any,
    ) -> tuple[bentoml.Tag, dict[str, t.Any]]:
        ...

    def make_tag(
        self,
        model_id: str | None = None,
        return_unused_kwargs: bool = False,
        trust_remote_code: bool = False,
        **attrs: t.Any,
    ) -> bentoml.Tag | tuple[bentoml.Tag, dict[str, t.Any]]:
        """Generate a ``bentoml.Tag`` from a given transformers model name.

        Note that this depends on your model to have a config class available.

        Args:
            model_name_or_path: The transformers model name or path to load the model from.
                                If it is a path, then `openllm_model_version` must be passed in as a kwarg.
            return_unused_kwargs: Whether to return unused kwargs. Defaults to False. If set, it will return a tuple
                                  of ``bentoml.Tag`` and a dict of unused kwargs.
            trust_remote_code: Whether to trust the remote code. Defaults to False.
            **attrs: Additional kwargs to pass to the ``transformers.AutoConfig`` constructor.
                    If your pass ``return_unused_kwargs=True``, it will be ignored.

        Returns:
            A tuple of ``bentoml.Tag`` and a dict of unused kwargs.
        """
        if model_id is None:
            model_id = self._model_id

        if "return_unused_kwargs" in attrs:
            logger.debug("Ignoring 'return_unused_kwargs' in 'generate_tag_from_model_name'.")
            attrs.pop("return_unused_kwargs", None)

        config, attrs = t.cast(
            "tuple[transformers.PretrainedConfig, dict[str, t.Any]]",
            transformers.AutoConfig.from_pretrained(
                model_id, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **attrs
            ),
        )
        name = convert_transformers_model_name(model_id)

        if os.path.exists(os.path.dirname(model_id)):
            # If the model_name_or_path is a path, we assume it's a local path,
            # then users must pass a version for this.
            model_version = attrs.pop("openllm_model_version", None)
            if model_version is None:
                logger.warning(
                    """\
            When passing a path, it is recommended to also pass 'openllm_model_version'
            into Runner/AutoLLM intialization.

            For example:

            >>> import openllm
            >>> runner = openllm.Runner('/path/to/fine-tuning/model', openllm_model_version='lora-version')

            Example with AutoLLM:

            >>> import openllm
            >>> model = openllm.AutoLLM.for_model('/path/to/fine-tuning/model', openllm_model_version='lora-version')

            No worries, OpenLLM will generate one for you. But for your own convenience, make sure to
            specify 'openllm_model_version'.
            """
                )
                model_version = bentoml.Tag.from_taglike(name).make_new_version().version
        else:
            model_version = getattr(config, "_commit_hash", None)
            if model_version is None:
                logger.warning(
                    "Given %s from '%s' doesn't contain a commit hash. We will generate"
                    " the tag without specific version.",
                    t.cast("type[transformers.PretrainedConfig]", config.__class__),
                    model_id,
                )
        tag = bentoml.Tag.from_taglike(f"{self.__llm_implementation__}-{name}:{model_version}")

        if return_unused_kwargs:
            return tag, attrs
        return tag

    def ensure_pretrained_exists(self) -> bentoml.Model:
        trust_remote_code = self._llm_attrs.pop("trust_remote_code", self.config.__openllm_trust_remote_code__)
        tag, kwds = self.make_tag(return_unused_kwargs=True, trust_remote_code=trust_remote_code, **self._llm_attrs)
        try:
            return bentoml.transformers.get(tag)
        except bentoml.exceptions.BentoMLException:
            logger.info("'%s' with tag (%s) not found, importing from HuggingFace Hub.", self.__class__.__name__, tag)
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

            return self.import_model(
                self._model_id,
                tag,
                *self._llm_args,
                tokenizer_kwds=tokenizer_kwds,
                trust_remote_code=trust_remote_code,
                **kwds,
            )

    @property
    def _bentomodel(self) -> bentoml.Model:
        if self.__llm_bentomodel__ is None:
            self.__llm_bentomodel__ = self.ensure_pretrained_exists()
        return self.__llm_bentomodel__

    @property
    def tag(self) -> bentoml.Tag:
        return self._bentomodel.tag

    @property
    def model(self) -> _M:
        """The model to use for this LLM. This shouldn't be set at runtime, rather let OpenLLM handle it."""
        # Run check for GPU
        trust_remote_code = self._llm_attrs.pop("trust_remote_code", self.config.__openllm_trust_remote_code__)
        self.config.check_if_gpu_is_available(implementation=self.__llm_implementation__)

        kwds = {k: v for k, v in self._llm_attrs.items() if not k.startswith("_tokenizer_")}

        if self.import_kwargs:
            kwds = {**{k: v for k, v in self.import_kwargs.items() if not k.startswith("_tokenizer_")}, **kwds}

        kwds["trust_remote_code"] = trust_remote_code
        if self.load_in_mha and "_pretrained_class" not in self._bentomodel.info.metadata:
            # This is a pipeline, provide a accelerator args
            kwds["accelerator"] = "bettertransformer"

        if self.__llm_model__ is None:
            if self.__openllm_custom_load__:
                self.__llm_model__ = self.load_model(self.tag, *self._llm_args, **kwds)
            else:
                self.__llm_model__ = self._bentomodel.load_model(*self._llm_args, **kwds)

        if (
            self.load_in_mha
            and all(i in self._bentomodel.info.metadata for i in ("_framework", "_pretrained_class"))
            and self._bentomodel.info.metadata["_framework"] == "torch"
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

    def to_runner(
        self,
        models: list[bentoml.Model] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, ModelSignatureDict | ModelSignature] | None = None,
        embedded: bool = False,
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
        """

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

            def __init_subclass__(cls, llm_type: str, identifying_params: dict[str, t.Any], **_: t.Any):
                cls.llm_type = llm_type
                cls.identifying_params = identifying_params

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

        def _wrapped_generate_run(__self: bentoml.Runner, *args: t.Any, **kwargs: t.Any) -> t.Any:
            """Wrapper for runner.generate.run() to handle the prompt and postprocessing.

            This will be used for LangChain API.

            Usage:
            ```python
            runner = openllm.Runner("dolly-v2", init_local=True)
            runner("What is the meaning of life?")
            ```
            """
            if len(args) > 1:
                raise RuntimeError("Only one positional argument is allowed for generate()")
            prompt = args[0] if len(args) == 1 else kwargs.pop("prompt", "")

            prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **kwargs)
            generated_result = __self.generate.run(prompt, **generate_kwargs)
            return self.postprocess_generate(prompt, generated_result, **postprocess_kwargs)

        # NOTE: returning the two langchain API's to the runner
        return types.new_class(
            inflection.camelize(self.config.__openllm_model_name__) + "Runner",
            (bentoml.Runner,),
            exec_body=lambda ns: ns.update(
                {
                    "llm_type": self.llm_type,
                    "identifying_params": self.identifying_params,
                    "llm": self,  # NOTE: self reference to LLM
                    "config": self.config,
                    "__call__": _wrapped_generate_run,
                    "__module__": f"openllm.models.{self.config.__openllm_model_name__}",
                    "__doc__": self.config.__openllm_env__.start_docstring,
                }
            ),
        )(
            types.new_class(
                inflection.camelize(self.config.__openllm_model_name__) + "Runnable",
                (_Runnable,),
                {
                    "SUPPORTED_RESOURCES": ("nvidia.com/gpu", "cpu")
                    if self.config.__openllm_requires_gpu__
                    else ("nvidia.com/gpu",),
                    "llm_type": self.llm_type,
                    "identifying_params": self.identifying_params,
                },
            ),
            name=name,
            models=models,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=bentoml_cattr.unstructure(method_configs),
            embedded=embedded,
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
        """
        prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **attrs)
        generated_result = self.generate(prompt, **generate_kwargs)
        return self.postprocess_generate(prompt, generated_result, **postprocess_kwargs)


def Runner(start_name: str, **attrs: t.Any) -> LLMRunner:
    """Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'

    Args:
        start_name: Supported model name from 'openllm models'
        init_local: Whether to init_local this given Runner. This is useful during development. (Default to False)
        **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs
                behaviour
    """
    init_local = attrs.pop("init_local", False)
    ModelEnv = openllm.utils.ModelEnv(start_name)
    if ModelEnv.get_framework_env() == "flax":
        runner = openllm.AutoFlaxLLM.create_runner(start_name, **attrs)
    elif ModelEnv.get_framework_env() == "tf":
        runner = openllm.AutoTFLLM.create_runner(start_name, **attrs)
    else:
        runner = openllm.AutoLLM.create_runner(start_name, **attrs)

    if init_local:
        runner.init_local(quiet=True)

    return runner
