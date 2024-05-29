# NOTE: pip install openai
import os, openai
from openai.types.chat import (
  ChatCompletionSystemMessageParam,
  ChatCompletionUserMessageParam,
)

STREAM = str(os.getenv('STREAM', False)).upper() in ['TRUE', '1', 'YES', 'Y', 'ON']

messages = [
  ChatCompletionSystemMessageParam(role='system', content='You will be the writing assistant that assume the tone of Ernest Hemmingway.'),
  ChatCompletionUserMessageParam(role='user', content='Write an essay on Nietzsche and absurdism.'),
]

def chat_stream(client: openai.OpenAI, model_id: str):
  for chunk in client.chat.completions.create(messages=messages, model=model_id, max_tokens=1024, stream=True):
    text = chunk.choices[0].delta.content
    if text: print(text, flush=True, end='')

def chat_one_shot(client: openai.OpenAI, model_id: str):
  completions = client.chat.completions.create(messages=messages, model=model_id, max_tokens=1024, stream=False)
  print(completions)


if __name__ == "__main__":
  client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')
  models = client.models.list()
  print('Models:', models.model_dump_json(indent=2))
  model_id = models.data[0].id
  chat_stream(client, model_id) if STREAM else chat_one_shot(client, model_id)
