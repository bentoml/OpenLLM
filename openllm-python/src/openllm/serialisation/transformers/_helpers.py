from __future__ import annotations
import copy
import typing as t

import openllm
import openllm_core

from bentoml._internal.models.model import ModelSignature
from openllm.serialisation.constants import FRAMEWORK_TO_AUTOCLASS_MAPPING
from openllm.serialisation.constants import HUB_ATTRS

if t.TYPE_CHECKING:
  import torch
  import transformers

  from transformers.models.auto.auto_factory import _BaseAutoModelClass

  from bentoml._internal.models.model import ModelSignaturesType
  from openllm_core._typing_compat import DictStrAny
  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T
else:
  transformers, torch = openllm_core.utils.LazyLoader('transformers', globals(), 'transformers'), openllm_core.utils.LazyLoader('torch', globals(), 'torch')

def process_config(model_id: str, trust_remote_code: bool, **attrs: t.Any) -> tuple[transformers.PretrainedConfig, DictStrAny, DictStrAny]:
  '''A helper function that correctly parse config and attributes for transformers.PretrainedConfig.

  Args:
    model_id: Model id to pass into ``transformers.AutoConfig``.
    trust_remote_code: Whether to trust_remote_code or not
    attrs: All possible attributes that can be processed by ``transformers.AutoConfig.from_pretrained

  Returns:
    A tuple of ``transformers.PretrainedConfig``, all hub attributes, and remanining attributes that can be used by the Model class.
  '''
  config = attrs.pop('config', None)
  # this logic below is synonymous to handling `from_pretrained` attrs.
  hub_attrs = {k: attrs.pop(k) for k in HUB_ATTRS if k in attrs}
  if not isinstance(config, transformers.PretrainedConfig):
    copied_attrs = copy.deepcopy(attrs)
    if copied_attrs.get('torch_dtype', None) == 'auto': copied_attrs.pop('torch_dtype')
    config, attrs = transformers.AutoConfig.from_pretrained(model_id, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_attrs, **copied_attrs)
  return config, hub_attrs, attrs

def infer_tokenizers_from_llm(__llm: openllm.LLM[t.Any, T], /) -> T:
  __cls = getattr(transformers, openllm_core.utils.first_not_none(__llm.config['tokenizer_class'], default='AutoTokenizer'), None)
  if __cls is None:
    raise ValueError(f'Cannot infer correct tokenizer class for {__llm}. Make sure to unset `tokenizer_class`')
  return __cls

def infer_autoclass_from_llm(llm: openllm.LLM[M, T], config: transformers.PretrainedConfig, /) -> _BaseAutoModelClass:
  if llm.config['trust_remote_code']:
    autoclass = 'AutoModelForSeq2SeqLM' if llm.config['model_type'] == 'seq2seq_lm' else 'AutoModelForCausalLM'
    if not hasattr(config, 'auto_map'):
      raise ValueError(f'Invalid configuraiton for {llm.model_id}. ``trust_remote_code=True`` requires `transformers.PretrainedConfig` to contain a `auto_map` mapping')
    # in case this model doesn't use the correct auto class for model type, for example like chatglm
    # where it uses AutoModel instead of AutoModelForCausalLM. Then we fallback to AutoModel
    if autoclass not in config.auto_map: autoclass = 'AutoModel'
    return getattr(transformers, autoclass)
  else:
    if type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING: idx = 0
    elif type(config) in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING: idx = 1
    else: raise openllm.exceptions.OpenLLMException(f'Model type {type(config)} is not supported yet.')
    return getattr(transformers, FRAMEWORK_TO_AUTOCLASS_MAPPING[llm.__llm_backend__][idx])

def check_unintialised_params(model: torch.nn.Module) -> None:
  unintialized = [n for n, param in model.named_parameters() if param.data.device == torch.device('meta')]
  if len(unintialized) > 0:
    raise RuntimeError(f'Found the following unintialized parameters in {model}: {unintialized}')

# NOTE: sync with bentoml/_internal/frameworks/transformers.py#make_default_signatures
def make_model_signatures(llm: openllm.LLM[M, T]) -> ModelSignaturesType:
  infer_fn: tuple[str, ...] = ('__call__',)
  default_config = ModelSignature(batchable=False)
  if llm.__llm_backend__ in {'pt', 'vllm'}:
    infer_fn += ('forward', 'generate', 'contrastive_search', 'greedy_search', 'sample', 'beam_search', 'beam_sample', 'group_beam_search', 'constrained_beam_search',)
  elif llm.__llm_backend__ == 'tf':
    infer_fn += ('predict', 'call', 'generate', 'compute_transition_scores', 'greedy_search', 'sample', 'beam_search', 'contrastive_search',)
  else:
    infer_fn += ('generate',)
  return {k: default_config for k in infer_fn}
