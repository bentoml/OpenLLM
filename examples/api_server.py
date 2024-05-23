from __future__ import annotations
import uuid, os
from typing import Any, AsyncGenerator, Dict, TypedDict, Union

from bentoml import Service
from bentoml.io import JSON, Text
from openllm import LLM

os.environ['IMPLEMENTATION'] = 'deprecated'

llm = LLM[Any, Any]('HuggingFaceH4/zephyr-7b-alpha', backend='vllm')


svc = Service('tinyllm', runners=[llm.runner])


class GenerateInput(TypedDict):
  prompt: str
  stream: bool
  sampling_params: Dict[str, Any]


@svc.api(
  route='/v1/generate',
  input=JSON.from_sample(
    GenerateInput(prompt='What is time?', stream=False, sampling_params={'temperature': 0.73, 'logprobs': 1})
  ),
  output=Text(content_type='text/event-stream'),
)
async def generate(request: GenerateInput) -> Union[AsyncGenerator[str, None], str]:
  n = request['sampling_params'].pop('n', 1)
  request_id = f'tinyllm-{uuid.uuid4().hex}'
  previous_texts = [[]] * n

  generator = llm.generate_iterator(request['prompt'], request_id=request_id, n=n, **request['sampling_params'])

  async def streamer() -> AsyncGenerator[str, None]:
    async for request_output in generator:
      for output in request_output.outputs:
        i = output.index
        previous_texts[i].append(output.text)
        yield output.text

  if request['stream']:
    return streamer()

  async for _ in streamer(): pass
  return ''.join(previous_texts[0])
