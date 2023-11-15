from __future__ import annotations
import argparse
import asyncio
import logging
import typing as t

import openllm

openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 384

Q = 'Answer the following question, step by step:\n{q}\nA:'
question = 'What is the meaning of life?'


async def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('question', default=question)

  if openllm.utils.in_notebook():
    args = parser.parse_args(args=[question])
  else:
    args = parser.parse_args()

  llm = openllm.LLM[t.Any, t.Any]('facebook/opt-2.7b')
  prompt = Q.format(q=args.question)

  logger.info('-' * 50, "Running with 'generate()'", '-' * 50)
  res = await llm.generate(prompt)
  logger.info('=' * 10, 'Response:', res)

  logger.info('-' * 50, "Running with 'generate()' with per-requests argument", '-' * 50)
  res = await llm.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
  logger.info('=' * 10, 'Response:', res)

  return 0


def _mp_fn(index: t.Any):  # type: ignore
  # For xla_spawn (TPUs)
  asyncio.run(main())
