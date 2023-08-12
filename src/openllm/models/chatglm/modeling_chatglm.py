from __future__ import annotations
import typing as t, openllm
if t.TYPE_CHECKING: import torch, transformers, torch.nn.functional as F
else: torch, transformers, F = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers"), openllm.utils.LazyLoader("F", globals(), "torch.nn.functional")

class ChatGLM(openllm.LLM["transformers.PreTrainedModel", "transformers.PreTrainedTokenizerFast"]):
  __openllm_internal__ = True

  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, num_beams: int | None = None, top_p: float | None = None, temperature: float | None = None, chat_history: list[tuple[str, str]] | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    prompt_text = ""
    if use_default_prompt_template and chat_history is not None:
      for i, (old_query, response) in enumerate(chat_history): prompt_text += f"[Round {i}]\n问:{old_query}\n答:{response}\n"
      prompt_text += f"[Round {len(chat_history)}]\n问:{prompt}\n答:"
    else: prompt_text = prompt
    postprocess_generate_kwargs = {"chat_history": chat_history if chat_history is not None else None}
    return prompt_text, {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "top_p": top_p, "temperature": temperature, **attrs}, postprocess_generate_kwargs
  def postprocess_generate(self, prompt: str, generation_result: tuple[str, list[tuple[str, str]]], *, chat_history: list[tuple[str, str]] | None = None, **attrs: t.Any) -> str:
    generated, history = generation_result
    if self.config.retain_history:
      if chat_history is None: raise ValueError("'retain_history' is True while there is no history provided.")
      chat_history.extend(history)
    return generated
  def generate(self, prompt: str, **attrs: t.Any) -> tuple[str, list[tuple[str, str]]]:
    with torch.inference_mode():
      self.model.eval()
      # Only use half precision if the model is not yet quantized
      if self.config.use_half_precision: self.model.half()
      return self.model.chat(self.tokenizer, prompt, generation_config=self.config.model_construct_env(**attrs).to_generation_config())
  def embeddings(self, prompts: list[str]) -> openllm.LLMEmbeddings:
    embeddings: list[list[float]] = []
    num_tokens = 0
    for prompt in prompts:
      input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
      with torch.inference_mode():
        outputs = self.model(input_ids, output_hidden_states=True)
        data = F.normalize(torch.mean(outputs.hidden_states[-1].transpose(0, 1), dim=0), p=2, dim=0)
        embeddings.append(data.tolist())
        num_tokens += len(input_ids[0])
    return openllm.LLMEmbeddings(embeddings=embeddings, num_tokens=num_tokens)
