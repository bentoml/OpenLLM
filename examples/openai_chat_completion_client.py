# NOTE: Make sure to install openai>1
import os, openai, typing as t
from openai.types.chat import (
  ChatCompletionMessageParam,
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
)

client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')

models = client.models.list()
print('Models:', models.model_dump_json(indent=2))
model = models.data[0].id

# Chat completion API
stream = str(os.getenv('STREAM', False)).upper() in ['TRUE', '1', 'YES', 'Y', 'ON']
messages: t.List[ChatCompletionMessageParam]= [
  ChatCompletionSystemMessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
  ChatCompletionUserMessageParam(role='user', content='Hi there!'),
  ChatCompletionAssistantMessageParam(role='assistant', content='Yes?'),
  ChatCompletionUserMessageParam(role='user', content='What is the meaning of life?'),
]
completions = client.chat.completions.create(messages=messages, model=model, max_tokens=128, stream=stream)

print(f'Chat completion result (stream={stream}):')
if stream:
  for chunk in completions:
    text = chunk.choices[0].delta.content
    if text:
      print(text, flush=True, end='')
else:
  print(completions)
