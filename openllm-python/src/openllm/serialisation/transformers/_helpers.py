from __future__ import annotations
import copy
import typing as t

import torch
import transformers

import openllm

from openllm.serialisation.constants import FRAMEWORK_TO_AUTOCLASS_MAPPING
from openllm.serialisation.constants import HUB_ATTRS


if t.TYPE_CHECKING:
  from transformers.models.auto.auto_factory import _BaseAutoModelClass

  from openllm_core._typing_compat import DictStrAny
  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T


def get_hash(config: transformers.PretrainedConfig) -> str:
  _commit_hash = getattr(config, '_commit_hash', None)
  if _commit_hash is None:
    raise ValueError(f'Cannot find commit hash in {config}')
  return _commit_hash


def process_config(
  model_id: str, trust_remote_code: bool, **attrs: t.Any
) -> tuple[transformers.PretrainedConfig, DictStrAny, DictStrAny]:
  """A helper function that correctly parse config and attributes for transformers.PretrainedConfig.

  Args:
    model_id: Model id to pass into ``transformers.AutoConfig``.
    trust_remote_code: Whether to trust_remote_code or not
    attrs: All possible attributes that can be processed by ``transformers.AutoConfig.from_pretrained

  Returns:
    A tuple of ``transformers.PretrainedConfig``, all hub attributes, and remanining attributes that can be used by the Model class.
  """
  config = attrs.pop('config', None)
  # this logic below is synonymous to handling `from_pretrained` attrs.
  hub_attrs = {k: attrs.pop(k) for k in HUB_ATTRS if k in attrs}
  if not isinstance(config, transformers.PretrainedConfig):
    copied_attrs = copy.deepcopy(attrs)
    if copied_attrs.get('torch_dtype', None) == 'auto':
      copied_attrs.pop('torch_dtype')
    config, attrs = transformers.AutoConfig.from_pretrained(
      model_id, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_attrs, **copied_attrs
    )
  return config, hub_attrs, attrs


def infer_autoclass_from_llm(llm: openllm.LLM[M, T], config: transformers.PretrainedConfig, /) -> _BaseAutoModelClass:
  if llm.trust_remote_code:
    autoclass = 'AutoModelForSeq2SeqLM' if llm.config['model_type'] == 'seq2seq_lm' else 'AutoModelForCausalLM'
    if not hasattr(config, 'auto_map'):
      raise ValueError(
        f'Invalid configuration for {llm.model_id}. ``trust_remote_code=True`` requires `transformers.PretrainedConfig` to contain a `auto_map` mapping'
      )
    # in case this model doesn't use the correct auto class for model type, for example like chatglm
    # where it uses AutoModel instead of AutoModelForCausalLM. Then we fallback to AutoModel
    if autoclass not in config.auto_map:
      autoclass = 'AutoModel'
    return getattr(transformers, autoclass)
  else:
    if type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING:
      idx = 0
    elif type(config) in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
      idx = 1
    else:
      raise openllm.exceptions.OpenLLMException(f'Model type {type(config)} is not supported yet.')
    return getattr(transformers, FRAMEWORK_TO_AUTOCLASS_MAPPING[llm.__llm_backend__][idx])


def check_unintialised_params(model: torch.nn.Module) -> None:
  unintialized = [n for n, param in model.named_parameters() if param.data.device == torch.device('meta')]
  if len(unintialized) > 0:
    raise RuntimeError(f'Found the following unintialized parameters in {model}: {unintialized}')
