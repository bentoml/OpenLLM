from __future__ import annotations

import pytest, subprocess, sys, openllm, bentoml, asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

SERVER_PORT = 53822


@pytest.mark.asyncio
async def test_openai_compatible(model_id: str):
  server = subprocess.Popen([sys.executable, '-m', 'openllm', 'start', model_id, '--port', str(SERVER_PORT)])
  await asyncio.sleep(5)
  with bentoml.SyncHTTPClient(f'http://127.0.0.1:{SERVER_PORT}', server_ready_timeout=60) as client:
    assert client.is_ready(30)

  try:
    client = AsyncOpenAI(api_key='na', base_url=f'http://127.0.0.1:{SERVER_PORT}/v1')
    serve_model = (await client.models.list()).data[0].id
    assert serve_model == openllm.utils.normalise_model_name(model_id)
    streamable = await client.chat.completions.create(
      model=serve_model,
      max_tokens=512,
      stream=False,
      messages=[
        ChatCompletionSystemMessageParam(
          role='system', content='You will be the writing assistant that assume the tone of Ernest Hemmingway.'
        ),
        ChatCompletionUserMessageParam(
          role='user', content='Comment on why Camus thinks we should revolt against life absurdity.'
        ),
      ],
    )
    assert streamable is not None
  finally:
    server.terminate()


@pytest.mark.asyncio
async def test_generate_endpoint(model_id: str):
  server = subprocess.Popen([sys.executable, '-m', 'openllm', 'start', model_id, '--port', str(SERVER_PORT)])
  await asyncio.sleep(5)

  try:
    with bentoml.SyncHTTPClient(f'http://127.0.0.1:{SERVER_PORT}', server_ready_timeout=60) as client:
      assert client.is_ready(30)

    async with bentoml.AsyncHTTPClient(f'http://127.0.0.1:{SERVER_PORT}', server_ready_timeout=60) as async_client:
      tasks = await asyncio.gather(
        async_client.generate_v1('Explain semiconductor to a 5 years old'),
        async_client.generate_v1(
          'Tell me more about Apple as a company', stop='technology', llm_config={'temperature': 0.5, 'top_p': 0.2}
        ),
      )
      assert len(tasks) == 2
  finally:
    server.terminate()
