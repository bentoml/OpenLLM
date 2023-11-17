from __future__ import annotations
import copy
import logging

import transformers

from openllm.serialisation.constants import HUB_ATTRS
from openllm_core.utils import get_disable_warnings, get_quiet_mode

logger = logging.getLogger(__name__)


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


def infer_autoclass_from_llm(llm, config, /):
  autoclass = 'AutoModelForSeq2SeqLM' if llm.config['model_type'] == 'seq2seq_lm' else 'AutoModelForCausalLM'
  if llm.trust_remote_code:
    if not hasattr(config, 'auto_map'):
      raise ValueError(
        f'Invalid configuration for {llm.model_id}. ``trust_remote_code=True`` requires `transformers.PretrainedConfig` to contain a `auto_map` mapping'
      )
    # in case this model doesn't use the correct auto class for model type, for example like chatglm
    # where it uses AutoModel instead of AutoModelForCausalLM. Then we fallback to AutoModel
    if autoclass not in config.auto_map:
      if not get_disable_warnings() and not get_quiet_mode():
        logger.warning(
          "OpenLLM failed to determine compatible Auto classes to load %s. Falling back to 'AutoModel'.\nTip: Make sure to specify 'AutoModelForCausalLM' or 'AutoModelForSeq2SeqLM' in your 'config.auto_map'. If your model type is yet to be supported, please file an issues on our GitHub tracker.",
          llm._model_id,
        )
      autoclass = 'AutoModel'
  return getattr(transformers, autoclass)
