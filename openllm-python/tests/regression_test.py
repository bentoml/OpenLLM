from __future__ import annotations

import pytest, subprocess, sys, openllm, asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam


@pytest.mark.asyncio
async def test_openai_compatible(server_port: str, model_id: str):
  server = subprocess.Popen([sys.executable, '-m', 'openllm', 'start', model_id, '--port', str(server_port)])

  asyncio.sleep(30)

  try:
    client = AsyncOpenAI(api_key='na', base_url=f'http://127.0.0.1:{server_port}/v1')
    serve_model = (await client.models.list()).data[0].id
    assert model_id == openllm.utils.normalise_model_name(model_id)
    streamable = await client.chat_completions.create(
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
