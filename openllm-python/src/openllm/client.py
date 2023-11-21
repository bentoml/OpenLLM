# fmt: off
import openllm_client as _client


def __dir__():return sorted(dir(_client))
def __getattr__(it):return getattr(_client, it)
