from __future__ import annotations
import os

import openai
import importlib.util

SUPPORT_LOGPROBS = str(os.getenv('ENABLE_LOGPROBS', default=importlib.util.find_spec('vllm') is not None)).upper() in ['TRUE', '1', 'YES', 'Y', 'ON']

openai.api_base = os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1'
openai.api_key = 'na'

MODEL = os.getenv('MODEL', 'na')  # XXX: CHANGE THIS TO THE MODEL USED AT $OPENLLM_ENDPOINT

print('Model metadata (/v1/models):\n')
print(openai.Model.list())

print('\nOne-shot completion (/v1/completions):\n')
print(openai.Completion.create(model=MODEL, prompt='Write a tagline for an ice cream shop.', max_tokens=12))

print('\nStreaming completion (/v1/completions):\n')
for chunk in openai.Completion.create(model=MODEL, prompt='Say this is a test', max_tokens=128, temperature=0.8, stream=True, logprobs=2 if SUPPORT_LOGPROBS else None):
  print(chunk.choices[0].text, flush=True, end='')

MESSAGES = [
  {
    'role': 'system',
    'content': 'You are acting as Ernest Hemmingway. You must answers questions that assume the identity of the writer. You must always return factual information and will not tolerate misleading ideology.',
  },
  {'role': 'user', 'content': 'Hi there!'},
  {'role': 'assistant', 'content': 'Yes?'},
  {'role': 'user', 'content': 'What is the meaning of life?'},
]
ARGS = dict(max_tokens=512, temperature=0.83, top_p=0.72, top_k=12, logprobs=2 if SUPPORT_LOGPROBS else None)

print('\nOne-shot chat completion (/v1/chat/completions):\n')
print(openai.ChatCompletion.create(model=MODEL, messages=MESSAGES, **ARGS))
print('\nStreaming chat completion (/v1/chat/completions):\n')
for chunk in openai.ChatCompletion.create(model=MODEL, messages=MESSAGES, stream=True, **ARGS):
  print(chunk.choices[0].delta.content, flush=True, end='')
