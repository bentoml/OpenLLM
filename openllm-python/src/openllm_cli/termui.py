from _openllm_tiny import _termui


def __dir__():
  return dir(_termui)


def __getattr__(name):
  return getattr(_termui, name)
