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
import typing as t
import openllm
from ..._llm import LLMEmbeddings
if t.TYPE_CHECKING:
    import torch, transformers, torch.nn.functional as F
else: torch, transformers, F = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers"), openllm.utils.LazyLoader("F", globals(), "torch.nn.functional")
class ChatGLM(openllm.LLM["transformers.PreTrainedModel", "transformers.PreTrainedTokenizerFast"]):
    __openllm_internal__ = True
    def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, num_beams: int | None = None, top_p: float | None = None, temperature: float | None = None, chat_history: list[str] | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        prompt_text = ""
        if use_default_prompt_template and chat_history is not None:
            for i, (old_query, response) in enumerate(chat_history): prompt_text += f"[Round {i}]\n问：{old_query}\n答：{response}\n"  # noqa: RUF001
            prompt_text += f"[Round {len(chat_history)}]\n问：{prompt}\n答："  # noqa: RUF001
        else: prompt_text = prompt
        postprocess_generate_kwargs = {"chat_history": chat_history if chat_history is not None else None}
        # NOTE: The rest of attrs should be kwargs for GenerationConfig
        generate_kwargs = {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "top_p": top_p, "temperature": temperature, **attrs}
        return prompt_text, generate_kwargs, postprocess_generate_kwargs
    def postprocess_generate(self, prompt: str, generation_result: tuple[str, list[tuple[str, str]]], *, chat_history: list[tuple[str, str]] | None = None, **attrs: t.Any):
        generated, history = generation_result
        if self.config.retain_history:
            assert chat_history is not None, "'retain_history' is True while there is no history provided."
            chat_history.extend(history)
        return generated
    def generate(self, prompt: str, **attrs: t.Any) -> tuple[str, list[tuple[str, str]]]:
        with torch.inference_mode():
            self.model.eval()
            # Only use half precision if the model is not yet quantized
            if self.config.use_half_precision: self.model.half()
            return self.model.chat(self.tokenizer, prompt, generation_config=self.config.model_construct_env(**attrs).to_generation_config())
    def embeddings(self, prompts: list[str]) -> LLMEmbeddings:
        embeddings: list[list[float]] = []
        num_tokens = 0
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                outputs = self.model(input_ids, output_hidden_states=True)
                data = F.normalize(torch.mean(outputs.hidden_states[-1].transpose(0, 1), dim=0), p=2, dim=0)
                embeddings.append(data.tolist())
                num_tokens += len(input_ids[0])
        return LLMEmbeddings(embeddings=embeddings, num_tokens=num_tokens)
