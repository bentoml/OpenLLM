# NOTE: Make sure to install openai>1
import os, openai
from openai.types.chat import (
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
)

client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')

models = client.models.list()
print('Models:', models.model_dump_json(indent=2))
model = models.data[0].id

# Chat completion API
stream = str(os.getenv('STREAM', False)).upper() in ['TRUE', '1', 'YES', 'Y', 'ON']
completions = client.chat.completions.create(messages=[
  ChatCompletionSystemMessageParam(role='system', content='You will be the writing assistant that assume the tone of Ernest Hemmingway.'),
  ChatCompletionUserMessageParam(role='user', content='Write an essay on Nietzsche and absurdism.'),
], model=model, max_tokens=1024, stream=stream)

print(f'Chat completion result (stream={stream}):')
if stream:
  for chunk in completions:
    text = chunk.choices[0].delta.content
    if text:
      print(text, flush=True, end='')
else:
  print(completions)
