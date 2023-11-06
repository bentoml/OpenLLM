from __future__ import annotations
import argparse
import asyncio
import json
import os

import aiohttp

import openllm

async def send_request(url, it, prompt, session, model, **attrs):
  headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
  config = openllm.AutoConfig.for_model(model).model_construct_env(**attrs).model_dump()
  data = {'prompt': prompt, 'llm_config': config, 'adapter_name': None}
  async with session.post(url, headers=headers, data=json.dumps(data)) as response:
    result = await response.text()
  print('-' * 10 + '\n\nreq:', it, ', prompt:', prompt, '\n\nGeneration:', result)

async def main(args: argparse.Namespace) -> int:
  endpoint = 'generate' if args.generate else 'generate_stream'
  url = f'{os.getenv("OPENLLM_ENDPOINT", "http://localhost:3000")}/v1/{endpoint}'
  # len=572
  prompts = [
      "Translate the following English text to French: 'Hello, how are you?'", "Summarize the plot of the book 'To Kill a Mockingbird.'",
      'Generate a list of 10 random numbers between 1 and 100.', 'What is the capital of France?', 'Write a poem about nature.', 'Convert 25 degrees Celsius to Fahrenheit.',
      'Describe the process of photosynthesis.', 'Tell me a joke.', 'List five famous scientists and their contributions to science.',
      'Write a short story about a detective solving a mystery.', 'Explain the theory of relativity.', 'Provide a brief history of the Roman Empire.',
      'Create a shopping list for a BBQ party.', "Write a movie review for the film 'Inception.'", 'Explain the concept of artificial intelligence.', 'Write a letter to your future self.',
      'Describe the life cycle of a butterfly.', 'List the top 10 tourist destinations in Europe.', 'Explain the principles of supply and demand.',
      'Create a menu for a vegetarian restaurant.', 'Write a haiku about the ocean.', 'Explain the importance of renewable energy sources.',
      'List the ingredients for making chocolate chip cookies.', 'Write a persuasive essay on the benefits of exercise.', 'Describe the cultural significance of the Taj Mahal.',
      'Explain the process of DNA replication.', 'Write a speech about the importance of education.', 'List the steps to start a small business.', 'Explain the concept of biodiversity.',
      'Create a playlist for a road trip.', 'Write a short biography of Albert Einstein.', 'Describe the impact of social media on society.', 'Explain the principles of good nutrition.',
      'List the 10 tallest mountains in the world.', 'Write a product review for a smartphone.', 'Create a workout routine for building muscle.', 'Explain the concept of climate change.',
      'Describe the life and achievements of Marie Curie.', 'List the ingredients for making a classic margarita.', 'Write a blog post about time management.',
      'Explain the process of cellular respiration.', 'Create a budget for a family vacation.', "Write a book summary for 'The Great Gatsby.'", 'Describe the history of the Internet.',
      'Explain the principles of effective communication.', 'List the top 10 historical landmarks in the world.', 'Write a love letter to someone special.',
      'Explain the concept of human rights.', 'Create a recipe for homemade pizza.', 'Write a movie script for a short film.', 'Describe the structure of the atom.',
      'List the 10 most influential artists of the 20th century.', 'Explain the process of mitosis.', 'Create a travel itinerary for a trip to Japan.',
      'Write a poem about the beauty of nature.', 'Explain the importance of environmental conservation.', 'List the essential items for a hiking trip.',
      'Write a short story set in a post-apocalyptic world.', 'Describe the history of the Olympic Games.', 'Explain the principles of democracy.',
      'Create a business plan for a tech startup.', 'Write a letter of recommendation for a colleague.', 'List the ingredients for a classic Caesar salad.',
      'Explain the concept of artificial neural networks.', 'Describe the life and work of Leonardo da Vinci.', 'List the 10 most popular tourist attractions in the United States.',
      'Write a persuasive speech on the dangers of smoking.', 'Explain the process of natural selection.', 'Create a menu for a fine dining restaurant.',
      'Write a poem about the beauty of the night sky.', 'Explain the importance of renewable energy.', 'List the necessary equipment for a camping trip.',
      'Write a short biography of William Shakespeare.', 'Describe the impact of social media on business marketing.', 'Explain the principles of project management.',
      'Create a playlist for a relaxing evening at home.', 'Write a blog post about the history of space exploration.', 'Explain the process of protein synthesis.',
      'List the 10 most famous landmarks in Europe.', 'Write a book review for a classic novel.', 'Describe the history of ancient Egypt.', 'Explain the concept of cultural diversity.',
      'Create a recipe for a gourmet sandwich.', 'Write a screenplay for a science fiction movie.', "Describe the structure of the Earth's atmosphere.",
      'List the 10 greatest inventions of all time.', 'Explain the process of meiosis.', 'Create a travel guide for a visit to Paris.', 'Write a poem about the changing seasons.',
      'Explain the importance of clean energy sources.', 'List the essential camping gear for a wilderness adventure.', 'Write a short story about a time-traveling adventure.',
      'Describe the history of the Renaissance.', 'Explain the principles of economics.', 'Create a business proposal for a new restaurant.',
      'Write a letter to your future self 10 years from now.', 'List the ingredients for a classic lasagna.', 'Explain the concept of machine learning.',
      'Describe the life and contributions of Martin Luther King Jr.', 'List the 10 most famous museums in the world.',
      'Write a persuasive essay on the importance of environmental conservation.', 'Explain the process of geological erosion.', 'Create a menu for a vegan cafe.',
      'Write a poem about the power of imagination.', 'Explain the significance of the Industrial Revolution.', 'List the items needed for a beach vacation.',
      'Write a short biography of Charles Darwin.', 'Describe the impact of globalization on cultures.', 'Explain the principles of time management.',
      'Create a playlist for a high-energy workout.', 'Write a blog post about the future of artificial intelligence.', 'Explain the process of DNA transcription.',
      'List the 10 most iconic landmarks in Asia.', 'Write a book summary for a popular self-help book.', 'Describe the history of the ancient Greeks.',
      'Explain the concept of social justice.', 'Create a recipe for a gourmet salad.', 'Write a screenplay for a romantic comedy movie.', "Describe the layers of the Earth's atmosphere.",
      'List the 10 most influential inventors in history.', 'Explain the process of plate tectonics.', 'Create a travel itinerary for a road trip across the USA.',
      'Write a poem about the wonders of the natural world.', 'Explain the importance of sustainable agriculture.', 'List the essential hiking gear for a mountain expedition.',
      'Write a short story about a futuristic dystopia.', 'Describe the history of the Middle Ages.',
      'Write a letter to your future self, offering reflections on personal growth, achievements, and aspirations, as well as words of encouragement and guidance for your future journey.',
      'List the ingredients for a classic chicken pot pie recipe, a beloved comfort food that combines tender chicken, vegetables, and a flaky pastry crust in a savory filling.',
      'Explain the concept of artificial neural networks and their pivotal role in machine learning and artificial intelligence applications, from image recognition to natural language processing.',
      'Describe the life and contributions of Albert Einstein, shedding light on his groundbreaking theories of relativity, his influence on modern physics, and his enduring legacy.',
      "List the 10 most iconic landmarks in Australia, celebrating the country's breathtaking natural landscapes, architectural wonders, and unique cultural sites.",
      'Write a persuasive speech on the importance of mental health awareness, advocating for destigmatization, access to mental health services, and compassionate support for those in need.',
      "Explain the process of plate tectonics and its profound impact on the Earth's geology, including the formation of continents, mountain ranges, and the movement of tectonic plates.",
      'Create a menu for a Mexican street food restaurant, featuring authentic and flavorful dishes that capture the essence of Mexican cuisine, from tacos to tamales.',
      'Write a poem about the beauty of a moonlit night, exploring themes of serenity, reflection, and the enchanting ambiance created by the gentle glow of the moon.',
      'Explain the significance of the Civil Rights Movement in the United States, highlighting the courageous individuals and leaders who paved the way for racial equality and justice.',
      'List the items needed for a camping trip in the wilderness, ensuring that outdoor enthusiasts are well-equipped for their adventure in the great outdoors.',
      'Write a short biography of Jane Austen, delving into her literary contributions and the enduring impact of her novels on literature, feminism, and societal norms.',
      'Describe the impact of social media on political activism, discussing its role in mobilizing movements, shaping public discourse, and bringing about political change.',
      'Explain the principles of effective leadership and management in the context of modern organizations, emphasizing the qualities and skills that define successful leaders.',
      'Create a playlist for a high-energy workout session, curating motivating songs that provide the perfect soundtrack for a productive fitness routine.',
      'Write a blog post about the future of artificial intelligence, exploring its potential applications in healthcare, transportation, and various industries that stand to be transformed.',
      'Explain the process of DNA replication, shedding light on the intricate mechanism by which genetic information is faithfully duplicated during cell division.',
      'List the 10 most famous landmarks in Asia, offering insights into their historical and cultural significance as well as their unique architectural features.',
      'Write a book review for a contemporary novel that captivated your imagination, sharing your thoughts on the plot, characters, and overall impact of the book.',
      'Describe the history of ancient Egypt, offering a glimpse into the rich civilization that thrived along the banks of the Nile River, from monumental architecture to hieroglyphic writing.',
      'Explain the concept of cultural diversity, emphasizing the importance of embracing and celebrating diverse cultures and perspectives in a globalized world.',
      'Create a recipe for a gourmet pasta dish that tantalizes the taste buds with a harmonious blend of flavors and textures, elevating a classic dish to culinary excellence.',
      'Write a screenplay for a thrilling action movie set in a dystopian future, weaving together elements of suspense, adventure, and compelling characters in a post-apocalyptic world.',
      "Describe the structure of the Earth's inner core, exploring its composition and the geophysical processes that drive the Earth's magnetic field and seismic activity.",
      'List the 10 most significant technological innovations of the 21st century, from transformative advances in communication to breakthroughs in medical science and beyond.',
      "Explain the process of geological earthquakes, delving into the geological forces and factors that contribute to the occurrence of seismic events and their impact on the Earth's surface.",
      'Create a travel itinerary for a trip to Tokyo, Japan, offering recommendations for cultural experiences, sightseeing, and culinary delights in this vibrant metropolis.',
      'Write a poem about the beauty of a sunset over the ocean, capturing the awe-inspiring colors and the sense of tranquility that descends as the sun dips below the horizon.',
      'Explain the importance of conserving endangered species, highlighting the critical role these species play in maintaining ecosystem balance and the urgent need for conservation efforts.',
      "List the essential gear for a camping and hiking adventure in a pristine natural wilderness, ensuring that outdoor enthusiasts are well-prepared for their journey into nature's beauty.",
      'Write a short story about a group of explorers embarking on a thrilling quest to uncover a hidden treasure, navigating treacherous landscapes and solving ancient riddles along the way.',
      'Describe the history of the Byzantine Empire, exploring its cultural achievements, architectural marvels such as the Hagia Sophia, and its enduring influence on Eastern Europe and beyond.',
      'Explain the principles of ethical leadership, delving into the moral responsibilities of leaders in various domains and the impact of ethical leadership on organizations and society.',
      'Create a business proposal for a sustainable fashion brand committed to eco-friendly practices, ethical sourcing, and transparency in the fashion industry.',
      'Write a letter to your future self, offering reflections on personal growth, achievements, and aspirations, as well as words of encouragement and guidance for your future journey.',
      'List the ingredients for a classic chicken pot pie recipe, a beloved comfort food that combines tender chicken, vegetables, and a flaky pastry crust in a savory filling.',
      'Explain the concept of artificial neural networks and their pivotal role in machine learning and artificial intelligence applications, from image recognition to natural language processing.',
      'Describe the life and contributions of Albert Einstein, shedding light on his groundbreaking theories of relativity, his influence on modern physics, and his enduring legacy.',
      "List the 10 most iconic landmarks in Australia, celebrating the country's breathtaking natural landscapes, architectural wonders, and unique cultural sites.",
      'Write a persuasive speech on the importance of mental health awareness, advocating for destigmatization, access to mental health services, and compassionate support for those in need.',
      "Explain the process of plate tectonics and its profound impact on the Earth's geology, including the formation of continents, mountain ranges, and the movement of tectonic plates.",
      'Create a menu for a Mexican street food restaurant, featuring authentic and flavorful dishes that capture the essence of Mexican cuisine, from tacos to tamales.',
      'Write a poem about the beauty of a moonlit night, exploring themes of serenity, reflection, and the enchanting ambiance created by the gentle glow of the moon.',
      'Explain the significance of the Civil Rights Movement in the United States, highlighting the courageous individuals and leaders who paved the way for racial equality and justice.',
      'List the items needed for a camping trip in the wilderness, ensuring that outdoor enthusiasts are well-equipped for their adventure in the great outdoors.',
      'Write a short biography of Jane Austen, delving into her literary contributions and the enduring impact of her novels on literature, feminism, and societal norms.',
      'Describe the impact of social media on political activism, discussing its role in mobilizing movements, shaping public discourse, and bringing about political change.',
      'Explain the principles of effective leadership and management in the context of modern organizations, emphasizing the qualities and skills that define successful leaders.',
      'Create a playlist for a high-energy workout session, curating motivating songs that provide the perfect soundtrack for a productive fitness routine.',
      'Write a blog post about the future of artificial intelligence, exploring its potential applications in healthcare, transportation, and various industries that stand to be transformed.',
      'Explain the process of DNA replication, shedding light on the intricate mechanism by which genetic information is faithfully duplicated during cell division.',
      'List the 10 most famous landmarks in Asia, offering insights into their historical and cultural significance as well as their unique architectural features.',
      'Write a book review for a contemporary novel that captivated your imagination, sharing your thoughts on the plot, characters, and overall impact of the book.',
      'Describe the history of ancient Egypt, offering a glimpse into the rich civilization that thrived along the banks of the Nile River, from monumental architecture to hieroglyphic writing.',
      'Explain the concept of cultural diversity, emphasizing the importance of embracing and celebrating diverse cultures and perspectives in a globalized world.',
      'Create a recipe for a gourmet pasta dish that tantalizes the taste buds with a harmonious blend of flavors and textures, elevating a classic dish to culinary excellence.',
      'Write a screenplay for a thrilling action movie set in a dystopian future, weaving together elements of suspense, adventure, and compelling characters in a post-apocalyptic world.',
      "Describe the structure of the Earth's inner core, exploring its composition and the geophysical processes that drive the Earth's magnetic field and seismic activity.",
      'List the 10 most significant technological innovations of the 21st century, from transformative advances in communication to breakthroughs in medical science and beyond.',
      "Explain the process of geological earthquakes, delving into the geological forces and factors that contribute to the occurrence of seismic events and their impact on the Earth's surface.",
      'Create a travel itinerary for a trip to Tokyo, Japan, offering recommendations for cultural experiences, sightseeing, and culinary delights in this vibrant metropolis.',
      'Write a poem about the beauty of a sunset over the ocean, capturing the awe-inspiring colors and the sense of tranquility that descends as the sun dips below the horizon.',
      'Explain the importance of conserving endangered species, highlighting the critical role these species play in maintaining ecosystem balance and the urgent need for conservation efforts.',
      "List the essential gear for a camping and hiking adventure in a pristine natural wilderness, ensuring that outdoor enthusiasts are well-prepared for their journey into nature's beauty.",
      'Write a short story about a group of explorers embarking on a thrilling quest to uncover a hidden treasure, navigating treacherous landscapes and solving ancient riddles along the way.',
      'Describe the history of the Byzantine Empire, exploring its cultural achievements, architectural marvels such as the Hagia Sophia, and its enduring influence on Eastern Europe and beyond.',
      'Explain the principles of ethical leadership, delving into the moral responsibilities of leaders in various domains and the impact of ethical leadership on organizations and society.',
      'Create a business proposal for a sustainable fashion brand committed to eco-friendly practices, ethical sourcing, and transparency in the fashion industry.',
      'Write a letter to your future self, offering reflections on personal growth, achievements, and aspirations, as well as words of encouragement and guidance for your future journey.'
  ]
  async with aiohttp.ClientSession() as session:
    await asyncio.gather(*[send_request(url, it, prompt, session, 'llama', max_new_tokens=2048) for it, prompt in enumerate(prompts)])
  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--generate', default=False, action='store_true', help='Whether to test with stream endpoint.')
  args = parser.parse_args()
  raise SystemExit(asyncio.run(main(args)))
