import openai

openai.api_base = "http://localhost:3000/v1"
openai.api_key = "na"

response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt="Write a tagline for an ice cream shop.")

print(response)
