import os

import bentoml
from bentoml.io import Text
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenLLM

SAMPLE_INPUT = "What is the weather in San Francisco?"

MODEL_NAME = os.environ.get("MODEL_NAME", "dolly-v2")

llm = OpenLLM(model_name=MODEL_NAME, embedded=False)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
svc = bentoml.Service("langchain-openllm", runners=[llm.runner])


@svc.api(input=Text.from_sample(SAMPLE_INPUT), output=Text())
def chat(input_text: str):
    return agent.run(input_text)
