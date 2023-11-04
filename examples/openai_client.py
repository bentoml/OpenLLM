from __future__ import annotations
import os

import openai

openai.api_base = os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1'
openai.api_key = 'na'

MODEL = "facebook--opt-1.3b"

print('Model:', openai.Model.list())

print(openai.Completion.create(model=MODEL, prompt='Write a tagline for an ice cream shop.', max_tokens=256))

for chunk in openai.Completion.create(model=MODEL, prompt='Say this is a test', max_tokens=7, temperature=0, stream=True):
  print(chunk)

print(openai.ChatCompletion.create(model=MODEL, messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}], max_tokens=64))

for chunk in openai.ChatCompletion.create(model=MODEL,
                                          messages=[{
                                              'role': 'system',
                                              'content': 'You are a helpful assistant.'
                                          }, {
                                              'role': 'user',
                                              'content': 'Hello!'
                                          }],
                                          stream=True,
                                          max_tokens=64):
  print(chunk)
