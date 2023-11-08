# mypy: disable-error-code="call-arg,misc,attr-defined,type-abstract,type-arg,valid-type,arg-type"
from __future__ import annotations
import logging
import os
import typing as t

import _service_vars as svars
import orjson

import bentoml
import openllm

from bentoml.io import JSON
from bentoml.io import Text

logger = logging.getLogger(__name__)

llm = openllm.LLM[t.Any, t.Any](svars.model_id,
                                model_tag=svars.model_tag,
                                prompt_template=openllm.utils.first_not_none(os.getenv('OPENLLM_PROMPT_TEMPLATE'), None),
                                system_message=openllm.utils.first_not_none(os.getenv('OPENLLM_SYSTEM_MESSAGE'), None),
                                serialisation=openllm.utils.first_not_none(os.getenv('OPENLLM_SERIALIZATION'), 'safetensors'),
                                adapter_map=orjson.loads(svars.adapter_map),
                                trust_remote_code=openllm.utils.check_bool_env('TRUST_REMOTE_CODE', default=False))
llm_config = llm.config
svc = bentoml.Service(name=f"llm-{llm_config['start_name']}-service", runners=[llm.runner])

llm_model_class = openllm.GenerationInput.from_llm_config(llm_config)

@svc.api(route='/v1/generate', input=JSON.from_sample(llm_model_class.examples()), output=JSON.from_sample(openllm.GenerationOutput.examples()))
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
  return await llm.generate(**llm_model_class(**input_dict).model_dump())

@svc.api(route='/v1/generate_stream', input=JSON.from_sample(llm_model_class.examples()), output=Text(content_type='text/event-stream'))
async def generate_stream_v1(input_dict: dict[str, t.Any]) -> t.AsyncGenerator[str, None]:
  async for it in llm.generate_iterator(**llm_model_class(**input_dict).model_dump()):
    yield f'data: {it.model_dump_json()}\n\n'
  yield 'data: [DONE]\n\n'

_Metadata = openllm.MetadataOutput(timeout=llm_config['timeout'],
                                   model_name=llm_config['model_name'],
                                   backend=llm.__llm_backend__,
                                   model_id=llm.model_id,
                                   configuration=llm_config.model_dump_json().decode(),
                                   prompt_template=llm.runner.prompt_template,
                                   system_message=llm.runner.system_message)

@svc.api(route='/v1/metadata', input=Text(), output=JSON.from_sample(_Metadata.model_dump()))
def metadata_v1(_: str) -> openllm.MetadataOutput:
  return _Metadata

openllm.mount_entrypoints(svc, llm)  # HACK: This must always be the last line in this file, as we will do some MK for OpenAPI schema.
