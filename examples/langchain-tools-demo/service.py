# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.llms import OpenLLM

import bentoml
from bentoml.io import Text


SAMPLE_INPUT = "What is the weather in San Francisco?"

llm = OpenLLM(
    model_name="dolly-v2",
    model_id="databricks/dolly-v2-7b",
    embedded=False,
)
llm = OpenLLM(model_name="dolly-v2", embedded=False)
tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
svc = bentoml.Service("langchain-openllm", runners=[llm.runner])


@svc.api(input=Text.from_sample(SAMPLE_INPUT), output=Text())
def chat(input_text: str):
    return agent.run(input_text)
