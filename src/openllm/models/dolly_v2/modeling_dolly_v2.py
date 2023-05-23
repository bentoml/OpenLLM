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

import logging
import re
import typing as t

import bentoml

import openllm

from .configuration_dolly_v2 import (DEFAULT_PROMPT_TEMPLATE, END_KEY,
                                     RESPONSE_KEY, get_special_token_id)

if t.TYPE_CHECKING:
    import torch
    import transformers
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)


class DollyV2(openllm.LLM, _internal=True):
    default_model = "databricks/dolly-v2-3b"

    variants = ["databricks/dolly-v2-3b", "databricks/dolly-v2-7b", "databricks/dolly-v2-12b"]

    def import_model(self, pretrained: str, tag: bentoml.Tag, *args: t.Any, **kwargs: t.Any):
        return bentoml.transformers.save_model(
            str(tag),
            transformers.AutoModelForCausalLM.from_pretrained(
                pretrained, device_map="auto", torch_dtype=torch.bfloat16
            ),
            custom_objects={"tokenizer": transformers.AutoTokenizer.from_pretrained(pretrained, padding_size="left")},
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_length: int | None = None,
        do_sample: bool = True,
        temperature: float | None = None,
        top_k: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
        **kwargs: t.Any,
    ):
        """This is a implementation of InstructionTextGenerationPipeline from databricks."""
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )
        response_key_token_id = None
        end_key_token_id = None

        llm_config = self.config.with_options(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                eos_token_id = end_key_token_id
            except ValueError:
                pass

        prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device) if attention_mask is not None else None,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            generation_config=llm_config.to_generation_config(),
        )

        out_b = generated_sequence.shape[0]

        generated_sequence: list[list[int]] = (
            generated_sequence.reshape(in_b, out_b // in_b, *generated_sequence.shape[1:])[0].numpy().tolist()
        )
        records: list[dict[str, t.Any]] = []
        for sequence in generated_sequence:
            # The response will be set to this variable if we can identify it.
            decoded = None

            # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
            if response_key_token_id and end_key_token_id:
                # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                # prompt, we should definitely find it.  We will return the tokens found after this token.
                try:
                    response_pos = sequence.index(response_key_token_id)
                except ValueError:
                    logger.warning(f"Could not find response key {response_key_token_id} in: {sequence}")
                    response_pos = None

                if response_pos:
                    # Next find where "### End" is located.  The model has been trained to end its responses with this
                    # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                    # this token, as the response could be truncated.  If we don't find it then just return everything
                    # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                    try:
                        end_pos = sequence.index(end_key_token_id)
                    except ValueError:
                        end_pos = None

                    decoded = self.tokenizer.decode(sequence[response_pos + 1 : end_pos]).strip()

            if not decoded:
                # Otherwise we'll decode everything and use a regex to find the response and end.

                fully_decoded = self.tokenizer.decode(sequence)

                # The response appears after "### Response:".  The model has been trained to append "### End" at the
                # end.
                m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

                if m:
                    decoded = m.group(1).strip()
                else:
                    # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                    # return everything after "### Response:".
                    m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                    if m:
                        decoded = m.group(1).strip()
                    else:
                        logger.warning(f"Failed to find response in:\n{fully_decoded}")

            # If the full text is requested, then append the decoded text to the original instruction.
            # This technically isn't the full text, as we format the instruction in the prompt the model has been
            # trained on, but to the client it will appear to be the full text.
            if llm_config.return_full_text:
                decoded = f"{prompt_text}\n{decoded}"

            rec = {"generated_text": decoded}

            records.append(rec)

        return records
