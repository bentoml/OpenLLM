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

from typing import Any
from typing import Dict

from langchain.chains import LLMChain
from langchain.llms import OpenLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

import bentoml
from bentoml.io import JSON
from bentoml.io import Text


class Query(BaseModel):
    industry: str
    product_name: str
    keywords: list[str]
    llm_config: Dict[str, Any]


llm = OpenLLM(
    model_name="dolly-v2",
    model_id="databricks/dolly-v2-7b",
    embedded=False,
)
prompt = PromptTemplate(
    input_variables=["industry", "product_name", "keywords"],
    template="""
You are a Facebook Ads Copywriter with a strong background in persuasive
writing and marketing. You craft compelling copy that appeals to the target
audience's emotions and needs, peruading them to take action or make a
purchase. You are given the following context to create a facebook ad copy.
It should provide an attention-grabbing headline optimizied for capivating
leads and perusaive calls to action.

Industry: {industry}
Product: {product_name}
Keywords: {keywords}
Facebook Ads copy:
    """,
)
chain = LLMChain(llm=llm, prompt=prompt)

svc = bentoml.Service("fb-ads-copy", runners=[llm.runner])

SAMPLE_INPUT = Query(
    industry="SAAS",
    product_name="BentoML",
    keywords=["open source", "developer tool", "AI application platform", "serverless", "cost-efficient"],
    llm_config=llm.runner.config.model_dump(),
)


@svc.api(input=JSON.from_sample(sample=SAMPLE_INPUT), output=Text())
def generate(query: Query):
    return chain.run(
        {"industry": query.industry, "product_name": query.product_name, "keywords": ", ".join(query.keywords)}
    )
