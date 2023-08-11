
from __future__ import annotations

import bentoml
import openllm

model = "dolly-v2"

llm_config = openllm.AutoConfig.for_model(model)
llm_runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(name="llm-service", runners=[llm_runner])

@svc.on_startup
def download(_: bentoml.Context):
  llm_runner.download_model()

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def prompt(input_text: str) -> str:
  answer = await llm_runner.generate.async_run(input_text)
  return answer[0]["generated_text"]
