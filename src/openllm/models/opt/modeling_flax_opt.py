from __future__ import annotations
import logging, typing as t, bentoml, openllm
from openllm._prompt import process_prompt
from openllm.utils import generate_labels
from .configuration_opt import DEFAULT_PROMPT_TEMPLATE
if t.TYPE_CHECKING: import transformers
else: transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)
class FlaxOPT(openllm.LLM["transformers.TFOPTForCausalLM", "transformers.GPT2Tokenizer"]):
  __openllm_internal__ = True
  def import_model(self, *args: t.Any, trust_remote_code: bool = False, **attrs: t.Any) -> bentoml.Model:
    config, tokenizer = transformers.AutoConfig.from_pretrained(self.model_id), transformers.AutoTokenizer.from_pretrained(self.model_id, **self.llm_parameters[-1])
    tokenizer.pad_token_id = config.pad_token_id
    return bentoml.transformers.save_model(self.tag, transformers.FlaxAutoModelForCausalLM.from_pretrained(self.model_id, **attrs), custom_objects={"tokenizer": tokenizer}, labels=generate_labels(self))
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, num_return_sequences: int | None = None, repetition_penalty: float | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "num_return_sequences": num_return_sequences, "repetition_penalty": repetition_penalty}, {}
  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **attrs: t.Any) -> str:
    if len(generation_result) == 1: return generation_result[0]
    if self.config.format_outputs: return "Generated result:\n" + "\n -".join(generation_result)
    else: return "\n".join(generation_result)
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]: return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors="np"), do_sample=True, generation_config=self.config.model_construct_env(**attrs).to_generation_config()).sequences, skip_special_tokens=True)
