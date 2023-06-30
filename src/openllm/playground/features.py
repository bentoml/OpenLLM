"""
General instroduction to OpenLLM.

This script will demo a few features from OpenLLM:
- Usage of Auto class abstraction and run prediction with 'generate'
- Ability to set per-requests parameters
- Runner integration with BentoML

python -m openllm.playground.features
"""

from __future__ import annotations

import argparse
import logging
import typing as t

import openllm


openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 384

Q = "Answer the following question, step by step:\n{q}\nA:"
question = "What is the meaning of life?"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", default=question)

    args = parser.parse_args()

    model = openllm.AutoLLM.for_model("opt", model_id="facebook/opt-2.7b")
    prompt = Q.format(q=args.question)

    logger.info("-" * 50, "Running with 'generate()'", "-" * 50)
    res = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
    logger.info("=" * 10, "Response:", model.postprocess_generate(prompt, res))

    logger.info("-" * 50, "Running with 'generate()' with per-requests argument", "-" * 50)
    res = model.generate(prompt, num_return_sequences=3)
    logger.info("=" * 10, "Response:", model.postprocess_generate(prompt, res))

    logger.info("-" * 50, "Using Runner abstraction with runner.generate.run()", "-" * 50)
    r = openllm.Runner("opt", model_id="facebook/opt-350m", init_local=True)
    res = r.generate.run(prompt)
    logger.info("=" * 10, "Response:", r.llm.postprocess_generate(prompt, res))

    logger.info("-" * 50, "Using Runner abstraction with runner()", "-" * 50)
    res = r(prompt)
    logger.info("=" * 10, "Response:", r.llm.postprocess_generate(prompt, res))

    return 0


def _mp_fn(index: t.Any):  # noqa # type: ignore
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    raise SystemExit(main())
