import bentoml
import openllm_core

from runner import llm_runner
from typing import Dict, Any
from prompt import prompt_template, HVAC_FUNCS

svc = bentoml.Service('assistant', runners=[llm_runner])

@svc.api(input=bentoml.io.Text.from_sample("i feel a little bit cold"), output=bentoml.io.JSON())
async def query(input: str) -> Dict:
    try:
        input_prompt = prompt_template(HVAC_FUNCS)(input)
        
        print("** llm prompt:", input_prompt)
        async for output in llm_runner.vllm_generate.async_stream(input_prompt, request_id=openllm_core.utils.gen_random_uuid()):
            responses = output
        print("** llm response:", responses[0])
        return responses[0]
    except Exception as e:
        print("** llm prompt:", input_prompt)
        print(e)
        return {"error_message": "internal error"}