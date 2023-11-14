import os

import cohere
from cohere.responses.chat import StreamTextGeneration

co = cohere.Client(api_key='na', api_url=os.getenv('OPENLLM_ENDPOINT', 'http://localhost:3000') + '/cohere')

generation = co.generate(prompt='Write me a tag line for an ice cream shop.')
print('One shot generation:', generation.generations[0].text)

print('\nStreaming response: ', flush=True, end='')
for it in co.generate(prompt='Write me a tag line for an ice cream shop.', stream=True):
  print(it.text, flush=True, end='')

for it in co.chat(
  message="What is Epicurus's philosophy of life?",
  temperature=0.6,
  chat_history=[
    {'role': 'User', 'message': 'What is the meaning of life?'},
    {
      'role': 'Chatbot',
      'message': "Many thinkers have proposed theories about the meaning of life. \n\nFor instance, Jean-Paul Sartre believed that existence precedes essence, meaning that the essence, or meaning, of one's life arises after birth. Søren Kierkegaard argued that life is full of absurdity and that one must make one's own values in an indifferent world. Arthur Schopenhauer stated that one's life reflects one's will, and that the will (or life) is without aim, irrational, and full of pain. \n\nEarly thinkers such as John Locke, Jean-Jacques Rousseau and Adam Smith believed that humankind should find meaning through labour, property and social contracts. \n\nAnother way of thinking about the meaning of life is to focus on the pursuit of happiness or pleasure. Aristippus of Cyrene, a student of Socrates, founded an early Socratic school that emphasised one aspect of Socrates's teachings: that happiness is the end goal of moral action and that pleasure is the supreme good. Epicurus taught that the pursuit of modest pleasures was the greatest good, as it leads to tranquility, freedom from fear and absence of bodily pain. \n\nUltimately, the meaning of life is a subjective concept and what provides life with meaning differs for each individual.",
    },
  ],
  stream=True,
):
  if isinstance(it, StreamTextGeneration):
    print(it.text, flush=True, end='')
  else:
    print(f'\nGenerated object: {it}')
