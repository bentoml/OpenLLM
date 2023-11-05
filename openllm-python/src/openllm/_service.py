# mypy: disable-error-code="call-arg,misc,attr-defined,type-abstract,type-arg,valid-type,arg-type"
from __future__ import annotations
import logging
import os
import typing as t
import warnings

import _service_vars as svars
import orjson

import bentoml
import openllm

# The following warnings from bitsandbytes, and probably not that important for users to see
warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization')
warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization')
warnings.filterwarnings('ignore', message='The installed version of bitsandbytes was compiled without GPU support.')

logger = logging.getLogger(__name__)

model = svars.model
model_id = svars.model_id
adapter_map = svars.adapter_map
llm_config = openllm.AutoConfig.for_model(model)
llm = openllm.LLM[t.Any, t.Any](model_id,
                                llm_config=llm_config,
                                prompt_template=openllm.utils.first_not_none(os.getenv('OPENLLM_PROMPT_TEMPLATE'), getattr(llm_config, 'default_prompt_template', None)),
                                system_message=openllm.utils.first_not_none(os.getenv('OPENLLM_SYSTEM_MESSAGE'), getattr(llm_config, 'default_system_message', None)),
                                serialisation=openllm.utils.first_not_none(os.getenv('OPENLLM_SERIALIZATION'), default=llm_config['serialisation']),
                                adapter_map=orjson.loads(adapter_map))
svc = bentoml.Service(name=f"llm-{llm_config['start_name']}-service", runners=[llm.runner])

_GenerateJsonInput = bentoml.io.JSON.from_sample({'prompt': '', 'stop': [], 'llm_config': llm_config.model_dump(flatten=True)})

@svc.api(route='/v1/generate', input=_GenerateJsonInput, output=bentoml.io.JSON.from_sample(openllm.GenerationOutput.examples().unmarshal()))
async def generate_v1(input_dict: dict[str, t.Any]) -> openllm.GenerationOutput:
  qa_inputs = openllm.GenerateInput.from_llm_config(llm_config)(**input_dict)
  return await llm.generate(qa_inputs.prompt, **qa_inputs.llm_config.model_dump())

@svc.api(route='/v1/generate_stream', input=_GenerateJsonInput, output=bentoml.io.Text(content_type='text/event-stream'))
async def generate_stream_v1(input_dict: dict[str, t.Any]) -> t.AsyncGenerator[str, None]:
  qa_inputs = openllm.GenerateInput.from_llm_config(llm_config)(**input_dict)
  return llm.generate_iterator(qa_inputs.prompt, return_type='text', **qa_inputs.llm_config.model_dump())

@svc.api(route='/v1/metadata',
         input=bentoml.io.Text(),
         output=bentoml.io.JSON.from_sample({
             'model_id': llm.model_id,
             'timeout': 3600,
             'model_name': llm_config['model_name'],
             'backend': llm.runner.backend,
             'configuration': llm_config.model_dump(flatten=True),
             'prompt_template': llm.runner.prompt_template,
             'system_message': llm.runner.system_message,
         }))
def metadata_v1(_: str) -> openllm.MetadataOutput:
  return openllm.MetadataOutput(timeout=llm_config['timeout'],
                                model_name=llm_config['model_name'],
                                backend=llm_config['env']['backend_value'],
                                model_id=llm.model_id,
                                configuration=llm_config.model_dump_json().decode(),
                                prompt_template=llm.runner.prompt_template,
                                system_message=llm.runner.system_message)

# HACK: This must always be the last line in this file, as we will do some MK for OpenAPI schema.
openllm.mount_entrypoints_to_svc(svc, llm)
