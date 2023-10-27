from __future__ import annotations
import time

import openllm

def clientQuery(llm_endpoint, query):
    client_conn_start = time.time()
    client = openllm.client.HTTPClient(llm_endpoint)
    client_conn_end = time.time() - client_conn_start

    client_embed_start = time.time()
    final_embedding = client.query(query, verify=False, max_new_tokens=100)
    client_embed_end = time.time() - client_embed_start

    print('final_embeddings are:', final_embedding)
    print('client conn time: ',client_conn_end)
    print('client embed time: ',client_embed_end)

llm_endpoint = 'http://server-link'
query = 'Write an essay on the topic of climate change and its impact on the environment and society. Please provide a comprehensive analysis of the causes, effects, and potential solutions. This essay should be at least 2000 words in length and cover all relevant aspects of the issue.'
clientQuery(llm_endpoint,query)
