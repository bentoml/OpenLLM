from __future__ import annotations

import bentoml
import openllm

llm = openllm.LLM('facebook/opt-2.7b')

svc = bentoml.Service(name="llm-service", runners=[llm.runner])

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def prompt(input_text: str) -> str:
  generation = await llm.generate(input_text)
  return generation.outputs[0].text
