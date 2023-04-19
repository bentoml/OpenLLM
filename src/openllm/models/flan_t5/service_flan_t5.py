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

import bentoml

import openllm

framework = openllm.utils.get_framework_env("flan-t5")
if framework == "flax":
    klass = openllm.FlaxLLM
elif framework == "pt":
    klass = openllm.LLM
elif framework == "tf":
    klass = openllm.TFLLM
else:
    raise ValueError(f"Invalid framework {framework}")

model_runner = klass.create_runner("flan-t5")
tokenizer_runner = openllm.Tokenizer.create_runner("flan-t5")

svc = bentoml.Service(name=openllm.utils.generate_service_name(model_runner), runners=[model_runner, tokenizer_runner])


@svc.api(input=openllm.Prompt(default="flan-t5"), output=openllm.Prompt(default="flan-t5"))
async def generate(qa: openllm.schema.PromptInput) -> openllm.schema.PromptOutput:
    """Returns the generated text from given prompts."""
    llm_config = model_runner.llm_config.dict()

    return_tensors = "np" if framework == "flax" else framework
    input_tensor = await tokenizer_runner.async_run(qa.prompt, return_tensors=return_tensors)
    if framework == "flax":
        outputs = await model_runner.generate.async_run(input_tensor["input_ids"], **llm_config)
        responses = await tokenizer_runner.batch_decode.async_run(
            outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    else:
        outputs = await model_runner.generate.async_run(input_tensor.input_ids, **llm_config)
        responses = await tokenizer_runner.batch_decode.async_run(outputs, skip_special_tokens=True)
    return openllm.schema.PromptOutput(responses=responses, configuration=llm_config)


@svc.api(input=bentoml.io.JSON(pydantic_model=openllm.FlanT5Config), output=bentoml.io.JSON())
def update_llm_config(llm_config: openllm.FlanT5Config) -> str:
    """Update the LLM configuration."""
    object.__setattr__(model_runner, "llm_config", llm_config)
    return model_runner.llm_config.dict()
