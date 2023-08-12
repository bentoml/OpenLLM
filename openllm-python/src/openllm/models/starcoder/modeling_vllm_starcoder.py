from __future__ import annotations
import logging, typing as t, openllm
from .configuration_starcoder import EOD, FIM_INDICATOR, FIM_MIDDLE, FIM_PAD, FIM_PREFIX, FIM_SUFFIX
if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)
class VLLMStarCoder(openllm.LLM["vllm.LLMEngine", "transformers.GPT2TokenizerFast"]):
  __openllm_internal__ = True
  tokenizer_id = "local"
  def sanitize_parameters(self, prompt: str, temperature: float | None = None, top_p: float | None = None, max_new_tokens: int | None = None, repetition_penalty: float | None = None, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    fim_mode, prefix, suffix = FIM_INDICATOR in prompt, None, None
    if fim_mode:
      try: prefix, suffix = prompt.split(FIM_INDICATOR)
      except Exception as err: raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt") from err
      prompt_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    else: prompt_text = prompt
    # XXX: This value for pad_token_id is currently a hack, need more investigate why the
    # default starcoder doesn't include the same value as santacoder EOD
    return prompt_text, {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty, "pad_token_id": 49152, **attrs}, {}
