# mypy: disable-error-code="name-defined,attr-defined"
from __future__ import annotations
import abc
import asyncio
import gc
import inspect
import logging
import os
import types
import typing as t

import attr
import fs.path
import inflection
import orjson

from huggingface_hub import hf_hub_download

import bentoml
import openllm
import openllm_core

from bentoml._internal.models.model import ModelSignature
from openllm_core._configuration import FineTuneConfig
from openllm_core._configuration import LLMConfig
from openllm_core._prompt import process_prompt
from openllm_core._schema import EmbeddingsOutput
from openllm_core._typing_compat import AdaptersMapping
from openllm_core._typing_compat import AdaptersTuple
from openllm_core._typing_compat import AdapterType
from openllm_core._typing_compat import DictStrAny
from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralQuantise
from openllm_core._typing_compat import LiteralSerialisation
from openllm_core._typing_compat import LiteralString
from openllm_core._typing_compat import LLMRunnable
from openllm_core._typing_compat import LLMRunner
from openllm_core._typing_compat import M
from openllm_core._typing_compat import ModelSignatureDict
from openllm_core._typing_compat import PeftAdapterOutput
from openllm_core._typing_compat import T
from openllm_core._typing_compat import TupleAny
from openllm_core._typing_compat import overload
from openllm_core.utils import DEBUG
from openllm_core.utils import MYPY
from openllm_core.utils import EnvVarMixin
from openllm_core.utils import LazyLoader
from openllm_core.utils import ReprMixin
from openllm_core.utils import apply
from openllm_core.utils import bentoml_cattr
from openllm_core.utils import codegen
from openllm_core.utils import ensure_exec_coro
from openllm_core.utils import first_not_none
from openllm_core.utils import generate_hash_from_file
from openllm_core.utils import is_peft_available
from openllm_core.utils import is_torch_available
from openllm_core.utils import normalize_attrs_to_model_tokenizer_pair
from openllm_core.utils import resolve_filepath
from openllm_core.utils import validate_is_path

from ._assign import make_llm_attributes
from ._quantisation import infer_quantisation_config
from .exceptions import ForbiddenAttributeError
from .exceptions import OpenLLMException
from .utils import infer_auto_class

if t.TYPE_CHECKING:

  import peft
  import torch
  import transformers
  import vllm

  from openllm_core._configuration import PeftType
  from openllm_core.utils.representation import ReprArgs
else:
  transformers = LazyLoader('transformers', globals(), 'transformers')
  torch = LazyLoader('torch', globals(), 'torch')
  peft = LazyLoader('peft', globals(), 'peft')

ResolvedAdaptersMapping = t.Dict[AdapterType, t.Dict[str, t.Tuple['peft.PeftConfig', str]]]

logger = logging.getLogger(__name__)

_object_setattr = object.__setattr__

def normalise_model_name(name: str) -> str:
  if validate_is_path(name): return os.path.basename(resolve_filepath(name))
  name = name.replace('/', '--')
  return inflection.dasherize(name)

# the below is similar to peft.utils.other.CONFIG_NAME
PEFT_CONFIG_NAME = 'adapter_config.json'

def resolve_peft_config_type(adapter_map: dict[str, str | None]) -> AdaptersMapping:
  '''Resolve the type of the PeftConfig given the adapter_map.

  This is similar to how PeftConfig resolve its config type.

  Args:
  adapter_map: The given mapping from either SDK or CLI. See CLI docs for more information.
  '''
  resolved: AdaptersMapping = {}
  _has_set_default = False
  for path_or_adapter_id, name in adapter_map.items():
    resolve_name = name
    if resolve_name is None:
      if _has_set_default: raise ValueError('Only one adapter can be set as default.')
      resolve_name = 'default'
      _has_set_default = True
    if os.path.isfile(os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)):
      config_file = os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)
    else:
      try:
        config_file = hf_hub_download(path_or_adapter_id, PEFT_CONFIG_NAME)
      except Exception as err:
        raise ValueError(f"Can't find '{PEFT_CONFIG_NAME}' at '{path_or_adapter_id}'") from err
    with open(config_file, 'r') as file:
      resolved_config = orjson.loads(file.read())
    # all peft_type should be available in PEFT_CONFIG_NAME
    _peft_type: AdapterType = resolved_config['peft_type'].lower()
    if _peft_type not in resolved: resolved[_peft_type] = ()
    resolved[_peft_type] += (_AdaptersTuple((path_or_adapter_id, resolve_name, resolved_config)),)
  return resolved

_reserved_namespace = {'config_class', 'model', 'tokenizer', 'import_kwargs'}

class LLMFunction(abc.ABC):
  @abc.abstractmethod
  def postprocess_generate(self, prompt: str, generation_result: t.Any, **attrs: t.Any) -> t.Any:
    '''This handler will postprocess generation results from LLM.generate and then output nicely formatted results (if the LLM decide to do so.).

    You can customize how the output of the LLM looks with this hook. By default, it is a simple echo.

    > [!NOTE]
    > This will be used from the client side.
    '''
    raise NotImplementedError

  @abc.abstractmethod
  def generate(self, prompt: str, **preprocess_generate_kwds: t.Any) -> t.Any:
    '''Text generation implementation for any given prompt.

    It takes the prompt and 'generation_kwargs'. The main implementation will parse all of kwargs
    correctly for you, so that subclass implementation don't have to repeat some of these boilercode.
    '''
    raise NotImplementedError

  @abc.abstractmethod
  def generate_iterator(self, prompt: str, /, **attrs: t.Any) -> t.Iterator[t.Any]:
    '''The iterator implementation of generate.

    This will be used for Token streaming and SSE support.

    Args:
      prompt: the input prompt
      **attrs: Relevant attributes to be pass to the stream generation implementation.

    Returns:
      An iterator of incoming token generation. It will returns a dictionary
    '''
    raise NotImplementedError

  def generate_one(self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any) -> t.Sequence[dict[t.Literal['generated_text'], str]]:
    '''The entrypoint for generating one prompt.

    This provides additional stop tokens for generating per token level. This is useful when running with agents, or initial streaming support.
    '''
    raise NotImplementedError

  def embeddings(self, prompts: list[str]) -> EmbeddingsOutput:
    '''The implementation for generating text embeddings from given prompt.

    It takes the prompt and output the embeddings for this given LLM.

    Returns:
        The embeddings for the given prompt.
    '''
    raise NotImplementedError

class LLMSerialisation(abc.ABC, t.Generic[M, T]):
  def import_model(self, *args: t.Any, trust_remote_code: bool, **attrs: t.Any) -> bentoml.Model:
    '''Import both model and tokenizer weights into as a BentoML models.

    Note that tokenizer attrs can be accessed via ``llm.llm_parameters``.

    ```python
    _, tokenizer_attrs = llm.llm_parameters
    ```

    By default, `model_decls` and `model_attrs` is already sanitised and concatenated into `args` and `attrs`
    '''
    raise NotImplementedError

  def load_model(self, *args: t.Any, **attrs: t.Any) -> M:
    '''This function can be implemented to override the default load_model behaviour.

    See falcon for example implementation. Tag can be accessed via ``self.tag``
    '''
    raise NotImplementedError

  def load_tokenizer(self, tag: bentoml.Tag, **attrs: t.Any) -> T:
    '''This function can be implemented to override how to load the tokenizer.

    See falcon for example implementation.
    '''
    raise NotImplementedError

class LLMInterface(LLMFunction, LLMSerialisation[M, T], abc.ABC):
  def llm_post_init(self) -> None:
    '''This function can be implemented if you need to initialized any additional variables that doesn't concern OpenLLM internals.
    By default, this will add `self.device` if the implementation is PyTorch.
    '''
    pass

  def sanitize_parameters(self, prompt: str, **attrs: t.Any) -> tuple[str, DictStrAny, DictStrAny]:
    '''This handler will sanitize all attrs and setup prompt text.

    It takes a prompt that is given by the user, attrs that can be parsed with the prompt.

    Returns a tuple of three items:
    - The attributes dictionary that can be passed into LLMConfig to generate a GenerationConfig
    - The attributes dictionary that will be passed into `self.postprocess_generate`.
    '''
    raise NotImplementedError

  @property
  def import_kwargs(self) -> tuple[DictStrAny, DictStrAny] | None:
    '''The default import kwargs to used when importing the model.

    This will be passed into 'openllm.LLM.import_model'.
    It returns two dictionaries: one for model kwargs and one for tokenizer kwargs.

    Returns:
        Optional tuple of model kwargs and tokenizer kwargs
    '''

  # NOTE: All fields below are attributes that can be accessed by users.
  config_class: t.Type[LLMConfig]
  '''The config class to use for this LLM. If you are creating a custom LLM, you must specify this class.'''
  device: 'torch.device'
  '''The device to be used for this LLM. If the implementation is 'pt', then it will be torch.device, else string.'''
  tokenizer_id: t.Union[t.Literal['local'], LiteralString]
  '''optional tokenizer_id for loading with vLLM if the model supports vLLM.'''

  # NOTE: The following will be populated by __init_subclass__, note that these should be immutable.
  __llm_backend__: LiteralBackend
  '''This is used to determine which framework implementation for this given LLM.

    Usually, this will inferred from class name, that follows the HuggingFace's naming convention:

    - `OPTForConditionalGeneration` -> `pt`
    - `TFOPTForConditionalGeneration` -> `tf`
    - `FlaxOPTForConditionalGeneration` -> `flax`

    For all VLLM backend: VLLMLlama -> `vllm`
    For all GGML backend: GGMLLlama -> `ggml`
    For all MLC backend: MLCLlama -> `mlc`
    '''
  __llm_model__: t.Optional[M]
  '''A reference to the actual model. Instead of access this directly, you should use `model` property instead.'''
  __llm_tokenizer__: t.Optional[T]
  '''A reference to the actual tokenizer. Instead of access this directly, you should use `tokenizer` property instead.'''
  __llm_adapter_map__: t.Optional[ResolvedAdaptersMapping]
  '''A reference to the the cached LoRA adapter mapping.'''

  __llm_supports_embeddings__: bool
  '''A boolean to determine whether models does implement ``LLM.embeddings``.'''
  __llm_supports_generate__: bool
  '''A boolean to determine whether models does implement ``LLM.generate``.'''
  __llm_supports_generate_one__: bool
  '''A boolean to determine whether models does implement ``LLM.generate_one``.'''
  __llm_supports_generate_iterator__: bool
  '''A boolean to determine whether models does implement ``LLM.generate_iterator``.'''

_DEFAULT_TOKENIZER = 'hf-internal-testing/llama-tokenizer'

_AdaptersTuple: type[AdaptersTuple] = codegen.make_attr_tuple_class('AdaptersTuple', ['adapter_id', 'name', 'config'])

@attr.define(slots=True, repr=False, init=False)
class LLM(LLMInterface[M, T], ReprMixin):
  if t.TYPE_CHECKING: __name__: str
  if t.TYPE_CHECKING and not MYPY:

    def __attrs_init__(self,
                       config: LLMConfig,
                       quantization_config: t.Optional[t.Union[transformers.BitsAndBytesConfig, transformers.GPTQConfig]],
                       quantize: t.Optional[LiteralQuantise],
                       model_id: str,
                       model_decls: TupleAny,
                       model_attrs: DictStrAny,
                       tokenizer_attrs: DictStrAny,
                       tag: bentoml.Tag,
                       adapters_mapping: t.Optional[AdaptersMapping],
                       model_version: t.Optional[str],
                       serialisation: LiteralSerialisation,
                       _local: bool,
                       **attrs: t.Any) -> None:
      '''Generated __attrs_init__ for openllm.LLM.'''

  config: LLMConfig
  '''The config instance to use for this LLM. This will be created based on config_class and available when initialising the LLM.'''
  quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | None
  '''Quantisation config for quantised model on the fly.'''
  _quantize: LiteralQuantise | None
  _model_id: str
  _model_decls: TupleAny
  _model_attrs: DictStrAny
  _tokenizer_attrs: DictStrAny
  _tag: bentoml.Tag
  _adapters_mapping: AdaptersMapping | None
  _model_version: str
  _serialisation: LiteralSerialisation
  _local: bool

  def __init_subclass__(cls: type[LLM[M, T]]) -> None:
    cd = cls.__dict__
    if cls.__name__.startswith('Flax'):
      cls.__llm_backend__, config_class = 'flax', openllm.AutoConfig.infer_class_from_name(cls.__name__[4:])
    elif cls.__name__.startswith('TF'):
      cls.__llm_backend__, config_class = 'tf', openllm.AutoConfig.infer_class_from_name(cls.__name__[2:])
    elif cls.__name__.startswith('VLLM'):
      cls.__llm_backend__, config_class = 'vllm', openllm.AutoConfig.infer_class_from_name(cls.__name__[4:])
    else:
      cls.__llm_backend__, config_class = 'pt', openllm.AutoConfig.infer_class_from_name(cls.__name__)
    if '__openllm_internal__' not in cd and 'config_class' not in cd:
      raise RuntimeError("Missing required key 'config_class'. Make sure to define it within the LLM subclass.")
    if '__openllm_internal__' in cd and 'config_class' not in cd: cls.config_class = config_class
    if 'tokenizer_id' not in cd and cls.__llm_backend__ == 'vllm': cls.tokenizer_id = _DEFAULT_TOKENIZER
    make_llm_attributes(cls)(cls)

  @overload
  def __getitem__(self, item: t.Literal['trust_remote_code']) -> bool:
    ...

  @overload
  def __getitem__(self, item: t.Literal['backend']) -> LiteralBackend:
    ...

  @overload
  def __getitem__(self, item: t.Literal['model']) -> M | None:
    ...

  @overload
  def __getitem__(self, item: t.Literal['tokenizer']) -> T | None:
    ...

  @overload
  def __getitem__(self, item: t.Literal['adapter_map']) -> ResolvedAdaptersMapping | None:
    ...

  @overload
  def __getitem__(self, item: t.Literal['supports_embeddings']) -> bool:
    ...

  @overload
  def __getitem__(self, item: t.Literal['supports_generate']) -> bool:
    ...

  @overload
  def __getitem__(self, item: t.Literal['supports_generate_one']) -> bool:
    ...

  @overload
  def __getitem__(self, item: t.Literal['supports_generate_iterator']) -> bool:
    ...

  def __getitem__(self, item: t.Union[LiteralString, t.Any]) -> t.Any:
    if item is None: raise TypeError(f"{self} doesn't understand how to index None.")
    item = inflection.underscore(item)
    internal_attributes = f'__llm_{item}__'
    if hasattr(self, internal_attributes): return getattr(self, internal_attributes)
    elif hasattr(self, item): return getattr(self, item)
    else: raise KeyError(item)

  @classmethod
  def from_pretrained(cls,
                      model_id: str | None = None,
                      model_version: str | None = None,
                      llm_config: LLMConfig | None = None,
                      *args: t.Any,
                      quantize: LiteralQuantise | None = None,
                      adapter_id: str | None = None,
                      adapter_name: str | None = None,
                      adapter_map: dict[str, str | None] | None = None,
                      quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | None = None,
                      serialisation: LiteralSerialisation = 'safetensors',
                      **attrs: t.Any) -> LLM[M, T]:
    '''Instantiate a pretrained LLM.

    ``LLM.from_pretrained`` follows the same design principle as HuggingFace's `from_pretrained` method, plus the following:

    ### Optimization options:

    > This is most notable during serving time.

    - quantize: quantize the model with the given quantization method. Currently supported int8, int4 quantization

    > Currently, the above two options are mutually exclusive.

    #### Quantisation options

    For customising options for quantisation config, ``openllm.LLM`` accepts all arbitrary arguments that is passed to ``transformers.BitsAndBytesConfig``
    plus ``quantize`` value. For example, for ``int8`` quantisation, specify the following:
    ```python
    model = openllm.AutoLLM.from_pretrained("opt", quantize='int8', llm_int8_enable_fp32_cpu_offload=False)
    ```

    ### Adapter options:

    > This is used in conjunction with the fine-tuning features

    - adapter_id: Optional [LoRA](https://arxiv.org/pdf/2106.09685.pdf) pretrained id or local path to apply to said model.
    - adapter_name: Optional name of the adapter to apply to said model. If not provided, it will be handled internally by OpenLLM.
    - adapter_map: optional dictionary of adapter_id to adapter_name. Note that this is mutually exclusive with adapter_id/adapter_name arguments.

    Args:
        model_id: The pretrained model to use. Defaults to None. If None, 'self.default_id' will be used.
                  > [!WARNING] If custom path is passed, make sure it contains all available file to construct
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
        quantization_config: The quantization config (`transformers.BitsAndBytesConfig` | `transformers.GPTQConfig`) to use. Note that this is mutually exclusive with `quantize`
        serialisation: Type of model format to save to local store. If set to 'safetensors', then OpenLLM will save model using safetensors.
                      Default behaviour is similar to ``safe_serialization=False``.
        adapter_id: The [LoRA](https://arxiv.org/pdf/2106.09685.pdf) pretrained id or local path to use for this LLM. Defaults to None.
        adapter_name: The adapter name to use for this LLM. Defaults to None.
        adapter_map: The adapter map to use for this LLM. Defaults to None. Note that this is mutually exclusive with adapter_id/adapter_name arguments.
        *args: The args to be passed to the model.
        **attrs: The kwargs to be passed to the model.
    '''
    cfg_cls = cls.config_class
    _local = False
    _model_id: str = first_not_none(model_id, os.environ.get(cfg_cls.__openllm_env__['model_id']), default=cfg_cls.__openllm_default_id__)
    if validate_is_path(_model_id): _model_id, _local = resolve_filepath(_model_id), True
    quantize = first_not_none(quantize, t.cast(t.Optional[LiteralQuantise], os.environ.get(cfg_cls.__openllm_env__['quantize'])), default=None)

    # quantization setup
    if quantization_config and quantize:
      raise ValueError("'quantization_config' and 'quantize' are mutually exclusive. Either customise your quantization_config or use the 'quantize' argument.")
    if quantization_config is None and quantize is not None:
      # in case users input `tokenizer` to __init__, default to the _model_id
      if quantize == 'gptq': attrs.setdefault('tokenizer', _model_id)
      quantization_config, attrs = infer_quantisation_config(cls, quantize, **attrs)

    # NOTE: LoRA adapter setup
    if adapter_map and adapter_id:
      raise ValueError(
          "'adapter_map' and 'adapter_id' are mutually exclusive. Either provide a 'adapter_map' ({adapter_id: adapter_name | None, ...}) or use the combination of adapter_id/adapter_name arguments. "
      )
    if adapter_map is None and adapter_id is not None: adapter_map = {adapter_id: adapter_name}
    if adapter_map is not None and not is_peft_available():
      raise RuntimeError("LoRA adapter requires 'peft' to be installed. Make sure to install OpenLLM with 'pip install \"openllm[fine-tune]\"'")
    if adapter_map: logger.debug('OpenLLM will apply the following adapters layers: %s', list(adapter_map))

    if llm_config is None:
      llm_config = cls.config_class.model_construct_env(**attrs)
      # The rests of the kwargs that is not used by the config class should be stored into __openllm_extras__.
      attrs = llm_config['extras']

    try:
      _tag = cls.generate_tag(_model_id, model_version)
      if _tag.version is None:
        raise ValueError(f'Failed to resolve the correct model version for {cfg_cls.__openllm_start_name__}')
    except Exception as err:
      raise OpenLLMException(f"Failed to generate a valid tag for {cfg_cls.__openllm_start_name__} with 'model_id={_model_id}' (lookup to see its traceback):\n{err}") from err

    return cls(*args,
               model_id=_model_id,
               llm_config=llm_config,
               quantization_config=quantization_config,
               _quantize=quantize,
               _model_version=_tag.version,
               _tag=_tag,
               _serialisation=serialisation,
               _local=_local,
               _adapters_mapping=resolve_peft_config_type(adapter_map) if adapter_map is not None else None,
               **attrs)

  @classmethod
  @apply(str.lower)
  def _generate_tag_str(cls, model_id: str, model_version: str | None) -> str:
    '''Generate a compliant ``bentoml.Tag`` from model_id.

    If model_id is a pretrained_id from HF, then it will have the following format: <backend>-<normalise_model_id>:<revision>
    If model_id contains the revision itself, then the same format above
    If model_id is a path, then it will be <backend>-<basename_of_path>:<generated_sha1> if model_version is not passesd, otherwise <backend>-<basename_of_path>:<model_version>

    > [!NOTE] here that the generated SHA1 for path cases is that it will be based on last modified time.

    Args:
        model_id: Model id for this given LLM. It can be pretrained weights URL, custom path.
        model_version: Specific revision for this model_id or custom version.

    Returns:
        ``str``: Generated tag format that can be parsed by ``bentoml.Tag``
    '''
    # specific branch for running in docker or kubernetes, this is very hacky,
    # and probably need a better way to support custom path
    if os.environ.get('BENTO_PATH') is not None: return ':'.join(fs.path.parts(model_id)[-2:])

    model_name = normalise_model_name(model_id)
    model_id, *maybe_revision = model_id.rsplit(':')
    if len(maybe_revision) > 0:
      if model_version is not None:
        logger.warning("revision is specified within 'model_id' (%s), and 'model_version=%s' will be ignored.", maybe_revision[0], model_version)
      return f'{cls.__llm_backend__}-{model_name}:{maybe_revision[0]}'

    tag_name = f'{cls.__llm_backend__}-{model_name}'
    if openllm_core.utils.check_bool_env('OPENLLM_USE_LOCAL_LATEST', False):
      return str(bentoml.models.get(f"{tag_name}{':'+model_version if model_version is not None else ''}").tag)
    if validate_is_path(model_id):
      model_id, model_version = resolve_filepath(model_id), first_not_none(model_version, default=generate_hash_from_file(model_id))
    else:
      from .serialisation.transformers._helpers import process_config
      model_version = getattr(
          process_config(model_id, trust_remote_code=cls.config_class.__openllm_trust_remote_code__, revision=first_not_none(model_version, default='main'))[0], '_commit_hash', None)
      if model_version is None:
        raise ValueError(f"Internal errors when parsing config for pretrained '{model_id}' ('commit_hash' not found)")
    return f'{tag_name}:{model_version}'

  @classmethod
  def generate_tag(cls, *param_decls: t.Any, **attrs: t.Any) -> bentoml.Tag:
    return bentoml.Tag.from_taglike(cls._generate_tag_str(*param_decls, **attrs))

  def __init__(self,
               *args: t.Any,
               model_id: str,
               llm_config: LLMConfig,
               quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | None,
               _quantize: LiteralQuantise | None,
               _model_version: str,
               _tag: bentoml.Tag,
               _serialisation: LiteralSerialisation,
               _local: bool,
               _adapters_mapping: AdaptersMapping | None,
               **attrs: t.Any,
               ):
    '''Initialize the LLM with given pretrained model.

    > [!WARNING]
    > To initializing any LLM, you should use `openllm.AutoLLM` or `openllm.LLM.from_pretrained` instead.
    > `__init__` initialization is only for internal use.

    > [!NOTE]
    > - *args to be passed to the model.
    > - **attrs will first be parsed to the AutoConfig, then the rest will be parsed to the import_model
    > - for tokenizer kwargs, it should be prefixed with _tokenizer_*

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
                    self.model_id, padding_side="left", **tokenizer_attrs
                )
            },
        )
    ```

    If your import model doesn't require customization, you can simply pass in `import_kwargs`
    at class level that will be then passed into The default `import_model` implementation.
    See ``openllm.DollyV2`` for example.

    ```python
    dolly_v2_runner = openllm.Runner(
        "dolly-v2", _tokenizer_padding_side="left", torch_dtype=torch.bfloat16, device_map="cuda"
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
        quantization_config: ``transformers.BitsAndBytesConfig`` configuration, or 'gptq' denoting this model to be loaded with GPTQ.
        *args: The args to be passed to the model.
        **attrs: The kwargs to be passed to the model.
    '''
    # low_cpu_mem_usage is only available for model
    # this is helpful on system with low memory to avoid OOM
    low_cpu_mem_usage = attrs.pop('low_cpu_mem_usage', True)
    if self.__llm_backend__ == 'pt':
      attrs.update({'low_cpu_mem_usage': low_cpu_mem_usage, 'quantization_config': quantization_config})
    model_kwds: DictStrAny = {}
    tokenizer_kwds: DictStrAny = {}
    if self.import_kwargs is not None: model_kwds, tokenizer_kwds = self.import_kwargs
    # set default tokenizer kwargs
    tokenizer_kwds.update({'padding_side': 'left', 'truncation_side': 'left'})

    # parsing tokenizer and model kwargs, as the hierachy is param pass > default
    normalized_model_kwds, normalized_tokenizer_kwds = normalize_attrs_to_model_tokenizer_pair(**attrs)
    # NOTE: Save the args and kwargs for latter load
    self.__attrs_init__(llm_config,
                        quantization_config,
                        _quantize,
                        model_id,
                        args, {
                            **model_kwds, **normalized_model_kwds
                        }, {
                            **tokenizer_kwds, **normalized_tokenizer_kwds
                        },
                        _tag,
                        _adapters_mapping,
                        _model_version,
                        _serialisation,
                        _local)

    self.llm_post_init()

  def __setattr__(self, attr: str, value: t.Any) -> None:
    if attr in _reserved_namespace:
      raise ForbiddenAttributeError(
          f'{attr} should not be set during runtime as these value will be reflected during runtime. Instead, you can create a custom LLM subclass {self.__class__.__name__}.')
    super().__setattr__(attr, value)

  @property
  def trust_remote_code(self) -> bool:
    return first_not_none(openllm_core.utils.check_bool_env('TRUST_REMOTE_CODE'), default=self.config['trust_remote_code'])

  @property
  def adapters_mapping(self) -> AdaptersMapping | None:
    return self._adapters_mapping

  @adapters_mapping.setter
  def adapters_mapping(self, value: AdaptersMapping) -> None:
    _object_setattr(self, '_adapters_mapping', value)

  @property
  def __repr_keys__(self) -> set[str]:
    return {'model_id', 'runner_name', 'config', 'adapters_mapping', 'tag'}

  def __repr_args__(self) -> ReprArgs:
    for k in self.__repr_keys__:
      if k == 'config': yield k, self.config.model_dump(flatten=True)
      else: yield k, getattr(self, k)
    yield 'backend', self.__llm_backend__

  @property
  def model_id(self) -> str:
    return self._model_id

  @property
  def runner_name(self) -> str:
    return f"llm-{self.config['start_name']}-runner"

  # NOTE: The section below defines a loose contract with langchain's LLM interface.
  @property
  def llm_type(self) -> str:
    return normalise_model_name(self._model_id)

  @property
  def identifying_params(self) -> DictStrAny:
    return {'configuration': self.config.model_dump_json().decode(), 'model_ids': orjson.dumps(self.config['model_ids']).decode()}

  @property
  def llm_parameters(self) -> tuple[tuple[tuple[t.Any, ...], DictStrAny], DictStrAny]:
    return (self._model_decls, self._model_attrs), self._tokenizer_attrs

  @property
  def tag(self) -> bentoml.Tag:
    return self._tag

  def save_pretrained(self) -> bentoml.Model:
    return openllm.import_model(self.config['start_name'],
                                model_id=self.model_id,
                                model_version=self._model_version,
                                backend=self.__llm_backend__,
                                quantize=self._quantize,
                                serialisation=self._serialisation)

  @property
  def _bentomodel(self) -> bentoml.Model:
    return openllm.serialisation.get(self, auto_import=True)

  def sanitize_parameters(self, prompt: str, **attrs: t.Any) -> tuple[str, DictStrAny, DictStrAny]:
    '''This handler will sanitize all attrs and setup prompt text.

    It takes a prompt that is given by the user, attrs that can be parsed with the prompt.

    Returns a tuple of three items:
    - The attributes dictionary that can be passed into LLMConfig to generate a GenerationConfig
    - The attributes dictionary that will be passed into `self.postprocess_generate`.
    '''
    return self.config.sanitize_parameters(prompt, **attrs)

  def postprocess_generate(self, prompt: str, generation_result: t.Any, **attrs: t.Any) -> t.Any:
    '''This handler will postprocess generation results from LLM.generate and then output nicely formatted results (if the LLM decide to do so.).

    You can customize how the output of the LLM looks with this hook. By default, it is a simple echo.

    > [!NOTE]
    > This will be used from the client side.
    '''
    if isinstance(generation_result, dict) and 'text' in generation_result: return generation_result['text']
    return self.config.postprocess_generate(prompt, generation_result, **attrs)

  @property
  def model(self) -> M:
    # NOTE: the signature of load_model here is the wrapper under _wrapped_load_model
    if self.__llm_model__ is None:
      model = self.load_model(*self._model_decls, **self._model_attrs)
      # If OOM, then it is probably you don't have enough VRAM to run this model.
      if self.__llm_backend__ == 'pt' and is_torch_available():
        loaded_in_kbit = getattr(model, 'is_loaded_in_8bit', False) or getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_quantized', False)
        if torch.cuda.is_available() and torch.cuda.device_count() == 1 and not loaded_in_kbit and not isinstance(model, transformers.Pipeline):
          try:
            model = model.to('cuda')
          except Exception as err:
            raise OpenLLMException(
                f'Failed to load {self} into GPU: {err}\nTip: If you run into OOM issue, maybe try different offload strategy. See https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/quantization#offload-between-cpu-and-gpu for more information.'
            ) from err
      self.__llm_model__ = model
    return self.__llm_model__

  @property
  def tokenizer(self) -> T:
    # NOTE: the signature of load_tokenizer here is the wrapper under _wrapped_load_tokenizer
    if self.__llm_tokenizer__ is None: self.__llm_tokenizer__ = self.load_tokenizer(**self._tokenizer_attrs)
    return self.__llm_tokenizer__

  def _transpose_adapter_mapping(self, inference_mode: bool = True, use_cache: bool = True) -> ResolvedAdaptersMapping:
    if self._adapters_mapping is None: raise ValueError('LoRA mapping is not set up correctly.')
    # early out if we already serialized everything.
    if use_cache and self.__llm_adapter_map__ is not None: return self.__llm_adapter_map__
    if not use_cache:
      logger.debug('Adapter mapping resolution will not be cached. This should only be used during training.')
    adapter_map: ResolvedAdaptersMapping = {k: {} for k in self._adapters_mapping}
    # this is a temporary check to accept the first option name as 'default'
    # then we will raise Error when the optional_name is set to None in next iteration.
    _converted_first_none = False
    for _adapter_type, _adapters_tuples in self._adapters_mapping.items():
      strategy = first_not_none(self.config['fine_tune_strategies'].get(_adapter_type),
                                default=FineTuneConfig(adapter_type=t.cast('PeftType', _adapter_type), llm_config_class=self.config_class))
      default_config = strategy.eval() if inference_mode else strategy.train()
      for adapter in _adapters_tuples:
        if not adapter.name and _converted_first_none:
          raise ValueError(f"{self.__class__.__name__} doesn't know how to resolve adapter_name None mapping: {adapter.adapter_id, adapter.config}")
        name = adapter.name
        if name is None:
          _converted_first_none = True
          name = 'default'
        peft_config = default_config.with_config(**adapter.config).to_peft_config() if name == 'default' else FineTuneConfig(
            adapter_type=t.cast('PeftType', _adapter_type), adapter_config=adapter.config, inference_mode=inference_mode, llm_config_class=self.config_class).to_peft_config()
        adapter_map[_adapter_type][name] = (peft_config, adapter.adapter_id)
    if self.__llm_adapter_map__ is None and use_cache: self.__llm_adapter_map__ = adapter_map
    return adapter_map

  def prepare_for_training(self, adapter_type: AdapterType = 'lora', use_gradient_checkpointing: bool = True, **attrs: t.Any) -> tuple[peft.PeftModel, T]:
    from peft import prepare_model_for_kbit_training
    peft_config = self.config['fine_tune_strategies'].get(adapter_type, FineTuneConfig(adapter_type=t.cast('PeftType', adapter_type),
                                                                                       llm_config_class=self.config_class)).train().with_config(**attrs).to_peft_config()
    wrapped_peft = peft.get_peft_model(prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        self.model, use_gradient_checkpointing=use_gradient_checkpointing), peft_config)
    if DEBUG: wrapped_peft.print_trainable_parameters()
    return wrapped_peft, self.tokenizer

  def apply_adapter(self, inference_mode: bool = True, adapter_type: AdapterType = 'lora', load_adapters: t.Literal['all'] | list[str] | None = None, use_cache: bool = True) -> M:
    '''Apply given LoRA mapping to the model. Note that the base model can still be accessed via self.model.get_base_model().'''
    if self.__llm_model__ is None: raise ValueError('Error: Model is not loaded correctly')
    # early out if _adapters_mapping is empty or it is already wrapped with peft.
    if not self._adapters_mapping: return self.__llm_model__
    if isinstance(self.__llm_model__, peft.PeftModel): return self.__llm_model__

    _mapping = self._transpose_adapter_mapping(inference_mode=inference_mode, use_cache=use_cache)
    if adapter_type not in _mapping:
      raise ValueError(f'Given adapter type {adapter_type} is not supported. Please choose from {list(_mapping.keys())}')
    adapter_mapping = _mapping[adapter_type]

    self.__llm_model__ = self._wrap_default_peft_model(adapter_mapping, inference_mode=inference_mode)
    # now we loop through the rest with add_adapter
    if len(adapter_mapping) > 0:
      for adapter_name, (_peft_config, _) in adapter_mapping.items():
        t.cast(peft.PeftModel, self.__llm_model__).add_adapter(adapter_name, _peft_config)

      # optionally load adapters. In case of multiple adapters, or on Runner,
      # we will need to set load_adapters='all'
      if load_adapters is not None:
        adapters_to_load = adapter_mapping.keys() if load_adapters == 'all' else load_adapters
        for adapter_name in adapters_to_load:
          _peft_config, _peft_model_id = adapter_mapping[adapter_name]
          t.cast(peft.PeftModel, self.__llm_model__).load_adapter(_peft_model_id, adapter_name=adapter_name, is_trainable=not inference_mode, **dict(_peft_config.to_dict()))

    return self.__llm_model__

  def _wrap_default_peft_model(self, adapter_mapping: dict[str, tuple[peft.PeftConfig, str]], inference_mode: bool) -> M:
    if self.__llm_model__ is None: raise ValueError('Error: Model is not loaded correctly')
    if isinstance(self.__llm_model__, peft.PeftModel): return self.__llm_model__
    if not isinstance(self.__llm_model__, transformers.PreTrainedModel):
      raise ValueError('Loading LoRA layers currently only runs on PyTorch models.')

    if 'default' not in adapter_mapping:
      raise ValueError("There is no 'default' mapping. Please check the adapter mapping and report this bug to the OpenLLM team.")
    default_config, peft_model_id = adapter_mapping.pop('default')

    # the below shared similar logics with `get_peft_model`
    # TODO: Support PromptLearningConfig
    if default_config.task_type not in peft.MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not isinstance(default_config, peft.PromptLearningConfig):
      logger.debug("Given task type '%s' is not supported by peft. Make sure the adapter is loaded manually before running inference.", default_config.task_type)
      model = peft.PeftModel(self.__llm_model__, default_config)
    else:
      # XXX: this is not ideal to serialize like this, maybe for fine-tune we will only support 0.4.0
      # onwards. For now, keep this logic here.
      peft_class = peft.MODEL_TYPE_TO_PEFT_MODEL_MAPPING[default_config.task_type]
      if default_config.base_model_name_or_path:
        kwargs: DictStrAny = {'is_trainable': not inference_mode}
        if 'config' in inspect.signature(peft_class.from_pretrained).parameters: kwargs['config'] = default_config
        else: kwargs.update(dict(default_config.to_dict().items()))
        # BUG: This hits during inference, need fixing
        model = peft_class.from_pretrained(self.__llm_model__, peft_model_id, **kwargs)
      else:
        # in this case, the given base_model_name_or_path is None. This will be hit during training
        model = peft_class(self.__llm_model__, default_config)
    return model

  # order of these fields matter here, make sure to sync it with
  # openllm.models.auto.factory.BaseAutoLLMClass.for_model
  def to_runner(self,
                models: list[bentoml.Model] | None = None,
                max_batch_size: int | None = None,
                max_latency_ms: int | None = None,
                scheduling_strategy: type[bentoml.Strategy] = openllm_core.CascadingResourceStrategy) -> LLMRunner[M, T]:
    '''Convert this LLM into a Runner.

    Args:
      models: Any additional ``bentoml.Model`` to be included in this given models.
      By default, this will be determined from the model_name.
      max_batch_size: The maximum batch size for the runner.
      max_latency_ms: The maximum latency for the runner.
      strategy: The strategy to use for this runner.
      embedded: Whether to run this runner in embedded mode.
      scheduling_strategy: Whether to create a custom scheduling strategy for this Runner.

    Returns:
      A generated LLMRunner for this LLM.

    > [!NOTE]: There are some difference between bentoml.models.get().to_runner() and LLM.to_runner():
    >
    > - 'name': will be generated by OpenLLM, hence users don't shouldn't worry about this. The generated name will be 'llm-<model-start-name>-runner' (ex: llm-dolly-v2-runner, llm-chatglm-runner)
    > - 'embedded': Will be disabled by default. There is no reason to run LLM in embedded mode.
    > - 'method_configs': The method configs for the runner will be managed internally by OpenLLM.
    '''
    models = models if models is not None else []

    if os.environ.get('BENTO_PATH') is None:
      # Hmm we should only add this if it is not in the container environment
      # BentoML sets BENTO_PATH so we can use this as switch logic here.
      try:
        models.append(self._bentomodel)
      except bentoml.exceptions.NotFound as err:
        raise RuntimeError(f'Failed to locate {self._bentomodel}:{err}') from None

    generate_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=False))
    embeddings_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=True, batch_dim=0))
    generate_iterator_sig = ModelSignature.from_dict(ModelSignatureDict(batchable=False))

    # NOTE: returning the two langchain API's to the runner
    return llm_runner_class(self)(llm_runnable_class(self, embeddings_sig, generate_sig, generate_iterator_sig),
                                  name=self.runner_name,
                                  embedded=False,
                                  models=models,
                                  max_batch_size=max_batch_size,
                                  max_latency_ms=max_latency_ms,
                                  method_configs=bentoml_cattr.unstructure({
                                      'embeddings': embeddings_sig,
                                      '__call__': generate_sig,
                                      'generate': generate_sig,
                                      'generate_one': generate_sig,
                                      'generate_iterator': generate_iterator_sig
                                  }),
                                  scheduling_strategy=scheduling_strategy)

  # NOTE: Scikit API
  def predict(self, prompt: str, **attrs: t.Any) -> t.Any:
    return self(prompt, **attrs)

  def __call__(self, prompt: str, **attrs: t.Any) -> t.Any:
    '''Returns the generation result and format the result.

    First, it runs `self.sanitize_parameters` to sanitize the parameters.
    The the sanitized prompt and kwargs will be pass into self.generate.
    Finally, run self.postprocess_generate to postprocess the generated result.

    This allows users to do the following:

    ```python
    llm = openllm.AutoLLM.for_model("dolly-v2")
    llm("What is the meaning of life?")
    ```
    '''
    prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **attrs)
    return self.postprocess_generate(prompt, self.generate(prompt, **generate_kwargs), **postprocess_kwargs)

  def generate_one(self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any) -> list[dict[t.Literal['generated_text'], str]]:
    max_new_tokens, encoded_inputs = preprocess_generate_kwds.pop('max_new_tokens', 200), self.tokenizer(prompt, return_tensors='pt').to(self.device)
    src_len, stopping_criteria = encoded_inputs['input_ids'].shape[1], preprocess_generate_kwds.pop('stopping_criteria', openllm.StoppingCriteriaList([]))
    stopping_criteria.append(openllm.StopSequenceCriteria(stop, self.tokenizer))
    result = self.tokenizer.decode(self.model.generate(encoded_inputs['input_ids'], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)[0].tolist()[src_len:])
    # Inference API returns the stop sequence
    for stop_seq in stop:
      if result.endswith(stop_seq): result = result[:-len(stop_seq)]
    return [{'generated_text': result}]

  def generate(self, prompt: str, **attrs: t.Any) -> t.List[t.Any]:
    # TODO: support different generation strategies, similar to self.model.generate
    for it in self.generate_iterator(prompt, **attrs):
      pass
    return [it]

  def generate_iterator(self,
                        prompt: str,
                        /,
                        *,
                        context_length: int | None = None,
                        echo: bool = False,
                        stream_interval: int = 2,
                        stop: str | t.Iterable[str] | None = None,
                        stop_token_ids: list[int] | None = None,
                        **attrs: t.Any) -> t.Iterator[t.Any]:
    # NOTE: encoder-decoder models will need to implement their own generate_iterator for now
    from ._generation import get_context_length
    from ._generation import is_partial_stop
    from ._generation import prepare_logits_processor

    len_prompt = len(prompt)
    config = self.config.model_construct_env(**attrs)
    if stop_token_ids is None: stop_token_ids = []
    stop_token_ids.append(self.tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(config)

    with torch.inference_mode():
      input_ids = self.tokenizer(prompt).input_ids

      if context_length is None: context_length = get_context_length(self.model.config)
      max_src_len = context_length - config['max_new_tokens'] - 1

      input_ids = input_ids[-max_src_len:]
      output_ids = list(input_ids)
      input_echo_len = len(input_ids)

      past_key_values = out = token = None
      finish_reason = None
      for i in range(config['max_new_tokens']):
        torch.cuda.synchronize()
        if i == 0:  # prefill
          out = self.model(torch.as_tensor([input_ids], device=self.device), use_cache=True)
        else:  # decoding
          out = self.model(torch.as_tensor([[token]], device=self.device), use_cache=True, past_key_values=past_key_values)
        logits = out.logits
        past_key_values = out.past_key_values
        torch.cuda.synchronize()

        if logits_processor:
          if config['repetition_penalty'] > 1.0:
            tmp_output_ids: t.Any = torch.as_tensor([output_ids], device=self.device)
          else:
            tmp_output_ids = None
          last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
          last_token_logits = logits[0, -1, :]

        # Switch to CPU by avoiding some bugs in mps backend.
        if self.device.type == 'mps': last_token_logits = last_token_logits.float().to('cpu')

        if config['temperature'] < 1e-5 or config['top_p'] < 1e-8:
          token = int(torch.argmax(last_token_logits))  # greedy
        else:
          probs = torch.softmax(last_token_logits, dim=-1)
          indices = torch.multinomial(probs, num_samples=2)
          token = int(indices.tolist()[0])
        output_ids.append(token)

        stopped = token in stop_token_ids

        # Yield the output tokens
        if i % stream_interval == 0 or i == config['max_new_tokens'] - 1 or stopped:
          if echo:
            tmp_output_ids = output_ids
            rfind_start = len_prompt
          else:
            tmp_output_ids = output_ids[input_echo_len:]
            rfind_start = 0
          output = self.tokenizer.decode(tmp_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False, clean_up_tokenization_spaces=True)

          partially_stopped = False
          if stop:
            if isinstance(stop, str):
              pos = output.rfind(stop, rfind_start)
              if pos != -1: output, stopped = output[:pos], True
              else: partially_stopped = is_partial_stop(output, stop)
            elif isinstance(stop, t.Iterable):
              for each_stop in stop:
                pos = output.rfind(each_stop, rfind_start)
                if pos != -1:
                  output, stopped = output[:pos], True
                  break
                else:
                  partially_stopped = is_partial_stop(output, each_stop)
                  if partially_stopped: break
            else: raise ValueError('Invalid stop field type.')

          # Prevent yielding partial stop sequence
          if not partially_stopped:
            yield {'text': output, 'usage': {'prompt_tokens': input_echo_len, 'completion_tokens': i, 'total_tokens': input_echo_len + i}, 'finish_reason': None}
        if stopped: break
      else: finish_reason = 'length'  # finish stream events
      if stopped: finish_reason = 'stop'
      yield {'text': output, 'usage': {'prompt_tokens': input_echo_len, 'completion_tokens': i, 'total_tokens': input_echo_len + i}, 'finish_reason': finish_reason}
    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()

@overload
def Runner(model_name: str, *, model_id: str | None = None, model_version: str | None = ..., init_local: t.Literal[False, True] = ..., **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  ...

@overload
def Runner(model_name: str,
           *,
           model_id: str = ...,
           model_version: str | None = ...,
           models: list[bentoml.Model] | None = ...,
           max_batch_size: int | None = ...,
           max_latency_ms: int | None = ...,
           method_configs: dict[str, ModelSignatureDict | ModelSignature] | None = ...,
           scheduling_strategy: type[bentoml.Strategy] | None = ...,
           **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  ...

@overload
def Runner(model_name: str,
           *,
           ensure_available: bool = ...,
           init_local: bool = ...,
           backend: LiteralBackend | None = None,
           llm_config: LLMConfig | None = None,
           **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  ...

@overload
def Runner(model_name: str,
           *,
           model_id: str | None = ...,
           model_version: str | None = ...,
           llm_config: LLMConfig | None = ...,
           quantize: LiteralQuantise | None = ...,
           adapter_id: str | None = ...,
           adapter_name: str | None = ...,
           adapter_map: dict[str, str | None] | None = ...,
           quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | None = None,
           serialisation: LiteralSerialisation = ...,
           **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  ...

def Runner(model_name: str,
           ensure_available: bool = False,
           init_local: bool = False,
           backend: LiteralBackend | None = None,
           llm_config: LLMConfig | None = None,
           **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  '''Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'.

  The behaviour of ensure_available that is synonymous to `AutoLLM.for_model` depends on `init_local`.
  By default, `ensure_available` is synonymous to `init_local`, meaning on the service when creating
  runner, it won't download the model. So before running your BentoML Service, you should create a `on_startup`
  hook to check download if you don't want to do it manually:

  ```python

  runner = openllm.Runner("dolly-v2")

  @svc.on_startup
  def download():
    runner.download_model()
  ```

  if `init_local=True` (For development workflow), it will also enable `ensure_available`.
  Default value of `ensure_available` is None. If set then use that given value, otherwise fallback to the aforementioned behaviour.

  Args:
    model_name: Supported model name from 'openllm models'
    ensure_available: If True, it will download the model if it is not available. If False, it will skip downloading the model.
                      If False, make sure the model is available locally.
    backend: The given Runner implementation one choose for this Runner. If `OPENLLM_BACKEND` is set, it will respect it.
    llm_config: Optional ``openllm.LLMConfig`` to initialise this ``openllm.LLMRunner``.
    init_local: If True, it will initialize the model locally. This is useful if you want to run the model locally. (Symmetrical to bentoml.Runner.init_local())
    **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs behaviour
  '''
  if llm_config is not None:
    attrs.update({
        'model_id': llm_config['env']['model_id_value'],
        'quantize': llm_config['env']['quantize_value'],
        'serialisation': first_not_none(os.environ.get('OPENLLM_SERIALIZATION'), attrs.get('serialisation'), default=llm_config['serialisation'])
    })

  backend = t.cast(LiteralBackend, first_not_none(backend, default=EnvVarMixin(model_name, backend=llm_config.default_backend() if llm_config is not None else 'pt')['backend_value']))
  if init_local: ensure_available = True
  runner = infer_auto_class(backend).create_runner(model_name, llm_config=llm_config, ensure_available=ensure_available, **attrs)
  if init_local: runner.init_local(quiet=True)
  return runner

def method_signature(sig: ModelSignature) -> ModelSignatureDict:
  return bentoml_cattr.unstructure(sig)

class SetAdapterOutput(t.TypedDict):
  success: bool
  message: str

def llm_runnable_class(self: LLM[M, T], embeddings_sig: ModelSignature, generate_sig: ModelSignature, generate_iterator_sig: ModelSignature) -> type[LLMRunnable[M, T]]:
  class _Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'amd.com/gpu', 'cpu')
    SUPPORTS_CPU_MULTI_THREADING = True
    backend = self.__llm_backend__

    def __init__(__self: _Runnable):
      # NOTE: The side effect of this line is that it will load the
      # imported model during runner startup. So don't remove it!!
      if not self.model: raise RuntimeError('Failed to load the model correctly (See traceback above)')
      if self.adapters_mapping is not None:
        logger.info('Applying LoRA to %s...', self.runner_name)
        self.apply_adapter(inference_mode=True, load_adapters='all')

    def set_adapter(__self: _Runnable, adapter_name: str) -> None:
      if self.__llm_adapter_map__ is None: raise ValueError('No adapters available for current running server.')
      elif not isinstance(self.model, peft.PeftModel): raise RuntimeError('Model is not a PeftModel')
      if adapter_name != 'default': self.model.set_adapter(adapter_name)
      logger.info('Successfully apply LoRA layer %s', adapter_name)

    @bentoml.Runnable.method(**method_signature(embeddings_sig))  # type: ignore
    def embeddings(__self: _Runnable, prompt: str | list[str]) -> t.Sequence[EmbeddingsOutput]:
      return [self.embeddings([prompt] if isinstance(prompt, str) else prompt)]

    @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
    def __call__(__self: _Runnable, prompt: str, **attrs: t.Any) -> list[t.Any]:
      adapter_name = attrs.pop('adapter_name', None)
      if adapter_name is not None: __self.set_adapter(adapter_name)
      return self.generate(prompt, **attrs)

    @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
    def generate(__self: _Runnable, prompt: str, **attrs: t.Any) -> list[t.Any]:
      adapter_name = attrs.pop('adapter_name', None)
      if adapter_name is not None: __self.set_adapter(adapter_name)
      return self.generate(prompt, **attrs)

    @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
    def generate_one(__self: _Runnable, prompt: str, stop: list[str], **attrs: t.Any) -> t.Sequence[dict[t.Literal['generated_text'], str]]:
      adapter_name = attrs.pop('adapter_name', None)
      if adapter_name is not None: __self.set_adapter(adapter_name)
      return self.generate_one(prompt, stop, **attrs)

    @bentoml.Runnable.method(**method_signature(generate_iterator_sig))  # type: ignore
    def generate_iterator(__self: _Runnable, prompt: str, **attrs: t.Any) -> t.Generator[str, None, str]:
      adapter_name = attrs.pop('adapter_name', None)
      if adapter_name is not None: __self.set_adapter(adapter_name)
      pre = 0
      for outputs in self.generate_iterator(prompt, request_id=openllm_core.utils.gen_random_uuid(), **attrs):
        output_text = outputs['text'].strip().split(' ')
        now = len(output_text) - 1
        if now > pre:
          yield ' '.join(output_text[pre:now]) + ' '
          pre = now
      yield ' '.join(output_text[pre:]) + ' '
      return ' '.join(output_text) + ' '

    @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
    def vllm_generate(__self: _Runnable, prompt: str, **attrs: t.Any) -> list[t.Any]:
      stop: str | t.Iterable[str] | None = attrs.pop('stop', None)
      stop_token_ids: list[int] | None = attrs.pop('stop_token_ids', None)
      adapter_name = attrs.pop('adapter_name', None)
      if adapter_name is not None: __self.set_adapter(adapter_name)
      request_id: str | None = attrs.pop('request_id', None)
      if request_id is None: raise ValueError('request_id must not be None.')

      if stop_token_ids is None: stop_token_ids = []
      stop_token_ids.append(self.tokenizer.eos_token_id)
      stop_: set[str] = set()
      if isinstance(stop, str) and stop != '': stop_.add(stop)
      elif isinstance(stop, list) and stop != []: stop_.update(stop)
      for tid in stop_token_ids:
        if tid: stop_.add(self.tokenizer.decode(tid))

      if self.config['temperature'] <= 1e-5: top_p = 1.0
      else: top_p = self.config['top_p']
      config = self.config.model_construct_env(stop=list(stop_), top_p=top_p, **attrs)
      sampling_params = config.to_sampling_config()

      async def loop() -> list[str]:
        async for request_output in t.cast('vllm.AsyncLLMEngine', self.model).generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
          pass
        return [output.text for output in request_output.outputs]

      try:
        return asyncio.run(loop())
      except RuntimeError:
        try:
          return ensure_exec_coro(loop())
        except Exception:
          raise

    @bentoml.Runnable.method(**method_signature(generate_iterator_sig))  # type: ignore
    async def vllm_generate_iterator(__self: _Runnable, prompt: str, **attrs: t.Any) -> t.AsyncGenerator[str, None]:
      # TODO: System prompt support
      pre = 0
      prompt = process_prompt(prompt, None, False)
      echo = attrs.pop('echo', False)
      stop: str | t.Iterable[str] | None = attrs.pop('stop', None)
      stop_token_ids: list[int] | None = attrs.pop('stop_token_ids', None)
      adapter_name = attrs.pop('adapter_name', None)
      if adapter_name is not None: __self.set_adapter(adapter_name)
      request_id: str | None = attrs.pop('request_id', None)
      if request_id is None: raise ValueError('request_id must not be None.')

      if stop_token_ids is None: stop_token_ids = []
      stop_token_ids.append(self.tokenizer.eos_token_id)
      stop_: set[str] = set()
      if isinstance(stop, str) and stop != '': stop_.add(stop)
      elif isinstance(stop, list) and stop != []: stop_.update(stop)
      for tid in stop_token_ids:
        if tid: stop_.add(self.tokenizer.decode(tid))

      if self.config['temperature'] <= 1e-5: top_p = 1.0
      else: top_p = self.config['top_p']
      config = self.config.model_construct_env(stop=list(stop_), top_p=top_p, **attrs)
      sampling_params = config.to_sampling_config()
      async for request_output in t.cast('vllm.AsyncLLMEngine', self.model).generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
        if echo: text_outputs = [prompt + output.text for output in request_output.outputs]
        else: text_outputs = [output.text for output in request_output.outputs]
        output_text = text_outputs[0]
        output_text = output_text.strip().split(' ')
        now = len(output_text) - 1
        if now > pre:
          yield ' '.join(output_text[pre:now]) + ' '
          pre = now
      yield ' '.join(output_text[pre:]) + ' '

  return types.new_class(self.__class__.__name__ + 'Runnable', (_Runnable,), {},
                         lambda ns: ns.update({
                             'SUPPORTED_RESOURCES': ('nvidia.com/gpu', 'amd.com/gpu', 'cpu'), '__module__': self.__module__, '__doc__': self.config['env'].start_docstring
                         }))

def llm_runner_class(self: LLM[M, T]) -> type[LLMRunner[M, T]]:
  def available_adapters(_: LLMRunner[M, T]) -> PeftAdapterOutput:
    if not is_peft_available():
      return PeftAdapterOutput(success=False, result={}, error_msg="peft is not available. Make sure to install: 'pip install \"openllm[fine-tune]\"'")
    if self.__llm_adapter_map__ is None:
      return PeftAdapterOutput(success=False, result={}, error_msg='No adapters available for current running server.')
    if not isinstance(self.model, peft.PeftModel):
      return PeftAdapterOutput(success=False, result={}, error_msg='Model is not a PeftModel')
    return PeftAdapterOutput(success=True, result=self.model.peft_config, error_msg='')

  def _wrapped_generate_run(__self: LLMRunner[M, T], prompt: str, **kwargs: t.Any) -> t.Any:
    '''Wrapper for runner.generate.run() to handle the prompt and postprocessing.

    This will be used for LangChain API.

    Usage:

    ```python
    runner = openllm.Runner("dolly-v2", init_local=True)
    runner("What is the meaning of life?")
    ```
    '''
    prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **kwargs)
    return self.postprocess_generate(prompt, __self.generate.run(prompt, **generate_kwargs), **postprocess_kwargs)

  def _wrapped_embeddings_run(__self: LLMRunner[M, T], prompt: str | list[str]) -> EmbeddingsOutput:
    '''``llm.embed`` is a light wrapper around runner.embeedings.run().

    Usage:

    ```python
    runner = openllm.Runner('llama', backend='pt')
    runner.embed("What is the meaning of life?")
    ```
    '''
    return __self.embeddings.run([prompt] if isinstance(prompt, str) else prompt)

  def _wrapped_repr_keys(_: LLMRunner[M, T]) -> set[str]:
    return {'config', 'llm_type', 'runner_methods', 'backend', 'llm_tag'}

  def _wrapped_repr_args(__self: LLMRunner[M, T]) -> ReprArgs:
    yield 'runner_methods', {method.name: {'batchable': method.config.batchable, 'batch_dim': method.config.batch_dim if method.config.batchable else None} for method in __self.runner_methods}
    yield 'config', self.config.model_dump(flatten=True)
    yield 'llm_type', __self.llm_type
    yield 'backend', self.__llm_backend__
    yield 'llm_tag', self.tag

  return types.new_class(self.__class__.__name__ + 'Runner', (bentoml.Runner,),
                         exec_body=lambda ns: ns.update({
                             'llm_type': self.llm_type,
                             'identifying_params': self.identifying_params,
                             'llm_tag': self.tag,
                             'llm': self,
                             'config': self.config,
                             'backend': self.__llm_backend__,
                             'peft_adapters': property(fget=available_adapters),
                             'download_model': self.save_pretrained,
                             '__call__': _wrapped_generate_run,
                             'embed': _wrapped_embeddings_run,
                             '__module__': self.__module__,
                             '__doc__': self.config['env'].start_docstring,
                             '__repr__': ReprMixin.__repr__,
                             '__repr_keys__': property(_wrapped_repr_keys),
                             '__repr_args__': _wrapped_repr_args,
                             'supports_embeddings': self['supports_embeddings'],
                             'supports_hf_agent': self['supports_generate_one'],
                             'has_adapters': self._adapters_mapping is not None
                         }))

__all__ = ['LLMRunner', 'LLMRunnable', 'Runner', 'LLM', 'llm_runner_class', 'llm_runnable_class', 'EmbeddingsOutput']
