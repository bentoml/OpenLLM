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

"""
General instroduction to OpenLLM.

This script will demo a few features from OpenLLM:
- Usage of Auto class abstraction and run prediction with 'generate'
- Ability to set per-requests parameters
- Runner integration with BentoML

python -m openllm.playground.general
"""

from __future__ import annotations

import logging

import openllm


openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 384

Q = "Answer the following question, step by step:\n{q}\nA:"
question = "What is the meaning of life?"

if __name__ == "__main__":
    model = openllm.AutoLLM.for_model("opt", model_id="facebook/opt-2.7b")
    prompt = Q.format(q=question)

    logger.info("-" * 50, "Running with 'generate()'", "-" * 50)
    res = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
    logger.info("=" * 10, "Response:", model.postprocess_generate(prompt, res))

    logger.info("-" * 50, "Running with 'generate()' with per-requests argument", "-" * 50)
    res = model.generate(prompt, num_return_sequences=3)
    logger.info("=" * 10, "Response:", model.postprocess_generate(prompt, res))

    logger.info("-" * 50, "Using Runner abstraction with runner.generate.run()", "-" * 50)
    del model

    r = openllm.Runner("opt", model_id="facebook/opt-350m", init_local=True)
    res = r.generate.run(prompt)
    logger.info("=" * 10, "Response:", r.llm.postprocess_generate(prompt, res))

    logger.info("-" * 50, "Using Runner abstraction with runner()", "-" * 50)
    res = r(prompt)
    logger.info("=" * 10, "Response:", r.llm.postprocess_generate(prompt, res))
