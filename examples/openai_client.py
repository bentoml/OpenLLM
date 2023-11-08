import os

# NOTE: Make sure to install openai>1
import openai
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

MODEL = os.getenv('MODEL', 'na')  # XXX: CHANGE THIS TO THE MODEL USED AT $OPENLLM_ENDPOINT
CLIENT = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')
MESSAGES = [
  ChatCompletionSystemMessageParam(
    role='system',
    content='You are acting as Ernest Hemmingway. You must answers questions that assume the identity of the writer. You must always return factual information and will not tolerate misleading ideology.',
  ),
  ChatCompletionUserMessageParam(role='user', content='Hi there!'),
  ChatCompletionAssistantMessageParam(role='assistant', content='Yes?'),
  ChatCompletionUserMessageParam(role='user', content='What is the meaning of life?'),
]


def completions() -> bool:
  print(CLIENT.completions.create(model=MODEL, prompt='Write a tagline for an ice cream shop.', max_tokens=64, temperature=0.83, logprobs=1))
  for chunk in CLIENT.completions.create(model=MODEL, prompt='Say this is a test', stream=True, max_tokens=64, temperature=0.83, logprobs=2):
    print(chunk.choices[0].text, flush=True, end='')
  return True


def chat_completions() -> bool:
  print(CLIENT.chat.completions.create(MESSAGES, MODEL, max_tokens=512, temperature=0.83, top_p=0.72))
  for chunk in CLIENT.chat.completions.create(MESSAGES, MODEL, stream=True, max_tokens=512, temperature=0.83, top_p=0.72):
    content = chunk.choices[0].delta.content
    if content:
      print(content, flush=True, end='')
  return True


def main() -> int:
  print('Model: %s' % CLIENT.models.list().model_dump())
  if not completions():
    return 1
  if not chat_completions():
    return 1
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
