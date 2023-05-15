from __future__ import annotations

import openllm

PROMPT_TEMPLATE = """\
Please use the following as context to answer the question: {context}
Question: {question}
Answer:"""

prompt = openllm.PromptTemplate.from_template(PROMPT_TEMPLATE)

client = openllm.client.create("0.0.0.0:3000", timeout=30)

print("----- Example query with prompt template -----")

print(
    client.query(prompt_template=prompt, context="This is a context", question="What is the answer?", temperature=0.48)
)

print("\n----- Example query with prompt template and return raw response -----")

print(
    client.query(
        prompt_template=prompt,
        context="This is a context",
        question="What is the answer?",
        return_raw_response=True,
        repetition_penalty=1.0,
    )
)

print("\n----- Example query using default template -----")

print(
    client.query(
        context="This is a useful context, and you should know about SQL injection.",
        question="What is SQL injection?",
        temperature=1.0,
    )
)
