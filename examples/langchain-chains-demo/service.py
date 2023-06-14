import bentoml
from bentoml.io import Text
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenLLM

llm = OpenLLM(
    model_name='dolly-v2',
    model_id='databricks/dolly-v2-7b',
    embedded=False,
)
prompt = PromptTemplate(
    input_variables=["company", "product"],
    template="What is a good name for {company} that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)

svc = bentoml.Service("langchain-openllm", runners=[llm.runner])

@svc.api(input=Text.from_sample(SAMPLE_INPUT), output=Text())
def chat(input_text: str):
    return agent.run(input_text)
