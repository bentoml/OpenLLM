from __future__ import annotations

import pytest, subprocess, sys, asyncio, openllm, bentoml
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

SERVER_PORT = 53822


@pytest.mark.asyncio
async def test_openai_compatible():
  model_id = 'casperhansen/llama-3-8b-instruct-awq'
  server = subprocess.Popen([sys.executable, '-m', 'openllm', 'start', model_id, '--port', str(SERVER_PORT)])
  await asyncio.sleep(10)
  with bentoml.SyncHTTPClient(f'http://127.0.0.1:{SERVER_PORT}', server_ready_timeout=12000) as client:
    assert client.is_ready(30)

  try:
    client = OpenAI(api_key='na', base_url=f'http://127.0.0.1:{SERVER_PORT}/v1')
    serve_model = client.models.list().data[0].id
    assert serve_model == openllm.utils.normalise_model_name(model_id)
    streamable = client.chat.completions.create(
      model=serve_model,
      max_tokens=128,
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
async def test_generate_endpoint():
  server = subprocess.Popen([
    sys.executable,
    '-m',
    'openllm',
    'start',
    'microsoft/Phi-3-mini-4k-instruct',
    '--trust-remote-code',
    '--port',
    str(SERVER_PORT),
  ])
  await asyncio.sleep(10)

  with bentoml.SyncHTTPClient(f'http://127.0.0.1:{SERVER_PORT}', server_ready_timeout=12000) as client:
    assert client.is_ready(30)

  try:
    client = openllm.HTTPClient(f'http://127.0.0.1:{SERVER_PORT}', api_version='v1')
    assert client.health()

    response = client.generate(
      'Tell me more about Apple as a company', stop='technology', llm_config={'temperature': 0.5, 'top_p': 0.2}
    )
    assert response is not None
  finally:
    server.terminate()
