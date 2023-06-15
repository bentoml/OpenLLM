import bentoml
import openllm
from bentoml.io import Text

model = "dolly-v2"

llm_config = openllm.AutoConfig.for_model(model)
llm_runner = openllm.Runner(model, llm_config=llm_config)

svc = bentoml.Service(name="llm-service", runners=[llm_runner])

@svc.api(input=Text(), output=Text())
async def prompt(input_text: str) -> str:
    answer = await llm_runner.generate(input_text)
    return answer
