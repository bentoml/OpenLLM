from __future__ import annotations
import os

import openai

openai.api_base = os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1'
openai.api_key = 'na'

print('Model:', openai.Model.list())

response = openai.Completion.create(model='gpt-3.5-turbo-instruct', prompt='Write a tagline for an ice cream shop.', max_tokens=256)

print(response)

for chunk in openai.Completion.create(model='gpt-3.5-turbo-instruct', prompt='Say this is a test', max_tokens=7, temperature=0, stream=True):
  print(chunk)

completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}])

print(completion)

completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hello!'}], stream=True)

for chunk in completion:
  print(chunk)
