from __future__ import annotations
import logging
import typing as t

import bentoml
import openllm
from openllm.utils import generate_labels
from openllm.utils import is_triton_available
if t.TYPE_CHECKING:
  import torch
  import transformers

logger = logging.getLogger(__name__)

def get_mpt_config(model_id_or_path: str,
                   max_sequence_length: int,
                   device: torch.device | str | int | None,
                   device_map: str | None = None,
                   trust_remote_code: bool = True) -> transformers.PretrainedConfig:
  import torch
  config = transformers.AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
  if hasattr(config, 'init_device') and device_map is None and isinstance(device, (str, torch.device)):
    config.init_device = str(device)
  if hasattr(config, 'attn_config') and is_triton_available(): config.attn_config['attn_impl'] = 'triton'
  else:
    logger.debug(
        "'triton' is not available, Flash Attention will use the default Torch implementation. For faster inference, make sure to install triton with 'pip install \"git+https://github.com/openai/triton.git#egg=triton&subdirectory=python\"'"
    )
  # setting max_seq_len
  config.max_seq_len = max_sequence_length
  return config

class MPT(openllm.LLM['transformers.PreTrainedModel', 'transformers.GPTNeoXTokenizerFast']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, 'torch_dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32}, {}

  def import_model(self, *args: t.Any, trust_remote_code: bool = True, **attrs: t.Any) -> bentoml.Model:
    import torch
    import transformers
    _, tokenizer_attrs = self.llm_parameters
    torch_dtype = attrs.pop('torch_dtype', torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    device_map = attrs.pop('device_map', None)
    attrs.pop('low_cpu_mem_usage', None)
    config = get_mpt_config(self.model_id, self.config.max_sequence_length, self.device, device_map=device_map, trust_remote_code=trust_remote_code)
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **tokenizer_attrs)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, config=config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, device_map=device_map, **attrs)
    try:
      return bentoml.transformers.save_model(self.tag, model, custom_objects={'tokenizer': tokenizer}, labels=generate_labels(self))
    finally:
      torch.cuda.empty_cache()

  def load_model(self, *args: t.Any, **attrs: t.Any) -> transformers.PreTrainedModel:
    import transformers
    torch_dtype = attrs.pop('torch_dtype', torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    device_map = attrs.pop('device_map', None)
    trust_remote_code = attrs.pop('trust_remote_code', True)
    config = get_mpt_config(self._bentomodel.path, self.config.max_sequence_length, self.device, device_map=device_map, trust_remote_code=trust_remote_code,)
    model = transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path,
                                                              config=config,
                                                              trust_remote_code=trust_remote_code,
                                                              torch_dtype=torch_dtype,
                                                              device_map=device_map,
                                                              **attrs)
    model.tie_weights()
    return model

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    llm_config = self.config.model_construct_env(**attrs)
    inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
    attrs = {
        'do_sample': False if llm_config['temperature'] == 0 else True,
        'eos_token_id': self.tokenizer.eos_token_id,
        'pad_token_id': self.tokenizer.pad_token_id,
        'generation_config': llm_config.to_generation_config()
    }
    with torch.inference_mode():
      if torch.cuda.is_available():
        with torch.autocast('cuda', torch.float16):  # type: ignore[attr-defined]
          generated_tensors = self.model.generate(**inputs, **attrs)
      else:
        generated_tensors = self.model.generate(**inputs, **attrs)
    return self.tokenizer.batch_decode(generated_tensors, skip_special_tokens=True)
