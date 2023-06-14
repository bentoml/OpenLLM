import bentoml
from bentoml.io import Text
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType            
from langchain.llms import OpenLLM

SAMPLE_INPUT = "What is the weather in San Francisco?"

llm = OpenLLM(
    model_name='dolly-v2',
    model_id='databricks/dolly-v2-7b',
    embedded=False,
)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
svc = bentoml.Service("langchain-openllm", runners=[llm.runner])

@svc.api(input=Text.from_sample(SAMPLE_INPUT), output=Text())
def chat(input_text: str):
    return agent.run(input_text)
