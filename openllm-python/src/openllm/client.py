def __dir__():
  import openllm_client as _client

  return sorted(dir(_client))


def __getattr__(it):
  import openllm_client as _client

  return getattr(_client, it)
