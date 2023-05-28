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

import bentoml
import transformers
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

import openllm

if t.TYPE_CHECKING:
    import torch
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")


class InvalidScoreLogitsProcessor(LogitsProcessor):
    """Ported from modeling_chatglm.py"""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLM(openllm.LLM):
    __openllm_internal__ = True
    __openllm_bettertransformer__ = False  # NOTE: disable bettertransformer for ChatGLM since it is already quantized

    default_model = "THUDM/chatglm-6b-int4"

    requirements = ["cpm_kernels", "sentencepiece"]

    variants = ["THUDM/chatglm-6b", "THUDM/chatglm-6b-int8", "THUDM/chatglm-6b-int4"]

    device = torch.device("cuda")

    def import_model(
        self, pretrained: str, tag: bentoml.Tag, *model_args: t.Any, tokenizer_kwds: dict[str, t.Any], **attrs: t.Any
    ) -> bentoml.Model:
        trust_remote_code = attrs.pop("trust_remote_code", True)
        return bentoml.transformers.save_model(
            str(tag),
            transformers.AutoModel.from_pretrained(pretrained, trust_remote_code=trust_remote_code),
            custom_objects={
                "tokenizer": transformers.AutoTokenizer.from_pretrained(
                    pretrained, trust_remote_code=trust_remote_code, **tokenizer_kwds
                )
            },
        )

    def preprocess_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        chat_history: list[str] | None = None,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any]]:
        prompt_text = ""
        if chat_history is not None:
            for i, (old_query, response) in enumerate(chat_history):
                prompt_text += f"[Round {i}]\n问：{old_query}\n答：{response}\n"
            prompt_text += f"[Round {len(chat_history)}]\n问：{prompt}\n答："
        else:
            prompt_text = prompt

        generation_config = self.config.model_construct_env(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            **attrs,
        ).model_dump(flatten=True)

        return prompt_text, generation_config

    def postprocess_parameters(
        self, prompt: str, generation_result: str, *, chat_history: list[tuple[str, str]] | None = None, **attrs: t.Any
    ):
        if self.config.retain_history:
            assert chat_history is not None, "'retain_history' is True while there is no history provided."
            chat_history.append((prompt, generation_result))
        return generation_result

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        **attrs: t.Any,
    ) -> str:
        self.model.eval()

        # Only use half precision if the model is not yet quantized
        if self.config.use_half_precision:
            self.model.half()

        self.model.cuda()

        logit_processor: list[LogitsProcessor] = LogitsProcessorList()
        logit_processor.append(InvalidScoreLogitsProcessor())

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.config.model_construct_env(
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                **attrs,
            ).to_generation_config(),
            logits_processor=logit_processor,
        )
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) :]
        response = self.tokenizer.decode(outputs)
        return self.model.process_response(response)
