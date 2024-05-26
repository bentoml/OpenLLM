from __future__ import annotations

import pytest, typing as t


@pytest.fixture(
  scope='function',
  name='model_id',
  params={
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'microsoft/Phi-3-mini-4k-instruct',
    'TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-AWQ',
  },
)
def fixture_model_id(request) -> t.Generator[str, None, None]:
  yield request.param
