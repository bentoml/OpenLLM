import cohere

client = cohere.Client(api_key='na', api_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/cohere')

generation = client.generate(prompt='Write me a tag line for an ice cream shop.')
print(generation)

for it in client.generate(prompt='Write me a tag line for an ice cream shop.', stream=True):
  print(it, flush=True, end='')
