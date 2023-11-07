from __future__ import annotations
import os

import openai
import importlib.util

SUPPORT_LOGPROBS = importlib.util.find_spec('vllm') is not None

openai.api_base = os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1'
openai.api_key = 'na'

MODEL = os.getenv('MODEL', 'na')  # XXX: CHANGE THIS TO THE MODEL USED AT $OPENLLM_ENDPOINT

print('\n' +'-'*50 + ' /v1/models ' + '-'*50 + "\n")
print(openai.Model.list())

print('\n' +'-'*50 + ' /v1/completions' + ' [stream=False] ' + '-'*50 + "\n")
print(openai.Completion.create(model=MODEL, prompt='Write a tagline for an ice cream shop.', max_tokens=128))

print('\n' +'-'*50 + ' /v1/completions' + ' [stream=True] ' + '-'*50 + "\n")
for chunk in openai.Completion.create(model=MODEL, prompt='Say this is a test', max_tokens=12, temperature=0.8, stream=True, logprobs=2 if SUPPORT_LOGPROBS else None):
  print(chunk)

print('\n' +'-'*50 + ' /v1/chat/completions' + ' [stream=False] ' + '-'*50 + "\n")
print(openai.ChatCompletion.create(model=MODEL, messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}], max_tokens=128, n=2, best_of=2))

print('\n' +'-'*50 + ' /v1/chat/completions' + ' [stream=True] ' + '-'*50 + "\n")
for chunk in openai.ChatCompletion.create(model=MODEL,
                                          messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}],
                                          stream=True,
                                          max_tokens=64):
  print(chunk)
