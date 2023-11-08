# NOTE: Make sure to install openai>1
import os, openai

client = openai.OpenAI(base_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/v1', api_key='na')
models = client.models.list()
print('Models:', models.model_dump_json(indent=2))
model = models.data[0].id

# Completion API
stream = str(os.getenv('STREAM', False)).upper() in ['TRUE', '1', 'YES', 'Y', 'ON']
completions = client.completions.create(prompt='Write me a tag line for an ice cream shop.', model=model, max_tokens=64, stream=stream)

print(f'Completion result (stream={stream}):')
if stream:
  for chunk in completions:
    text = chunk.choices[0].text
    if text:
      print(text, flush=True, end='')
else:
  print(completions)
