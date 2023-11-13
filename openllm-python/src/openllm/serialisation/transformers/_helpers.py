from __future__ import annotations
import copy
import typing as t

import transformers

from openllm.serialisation.constants import FRAMEWORK_TO_AUTOCLASS_MAPPING, HUB_ATTRS
from openllm_core.exceptions import OpenLLMException

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import M, T

  from ..._llm import LLM


def get_hash(config) -> str:
  _commit_hash = getattr(config, '_commit_hash', None)
  if _commit_hash is None:
    raise ValueError(f'Cannot find commit hash in {config}')
  return _commit_hash


def process_config(model_id, trust_remote_code, **attrs):
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


def infer_autoclass_from_llm(llm: LLM[M, T], config, /):
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
      raise OpenLLMException(f'Model type {type(config)} is not supported yet.')
    return getattr(transformers, FRAMEWORK_TO_AUTOCLASS_MAPPING[llm.__llm_backend__][idx])
