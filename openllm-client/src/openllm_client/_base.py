# mypy: disable-error-code="override,no-redef"
from __future__ import annotations
import abc
import functools
import logging
import typing as t

from http import HTTPStatus
from urllib.parse import urljoin

import attr
import httpx
import orjson

import openllm_core

from openllm_core._typing_compat import LiteralString
from openllm_core._typing_compat import overload
from openllm_core.utils import bentoml_cattr
from openllm_core.utils import ensure_exec_coro
from openllm_core.utils import is_transformers_available

from .benmin import AsyncClient as AsyncBentoClient
from .benmin import Client as BentoClient

if t.TYPE_CHECKING:
  import transformers

  from openllm_core._typing_compat import DictStrAny
  from openllm_core._typing_compat import LiteralBackend

logger = logging.getLogger(__name__)

@attr.define(slots=False, init=False)
class _ClientAttr:
  _address: str
  _timeout: float = attr.field(default=30)
  _api_version: str = attr.field(default='v1')

  def __init__(self, address: str, timeout: float = 30, api_version: str = 'v1'):
    self.__attrs_init__(address, timeout, api_version)

  @abc.abstractmethod
  def call(self, api_name: str, *args: t.Any, **attrs: t.Any) -> t.Any:
    raise NotImplementedError

  @abc.abstractmethod
  def _run_hf_agent(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    raise NotImplementedError

  @overload
  @abc.abstractmethod
  def query(self, prompt: str, *, return_response: t.Literal['processed'], **attrs: t.Any) -> str:
    ...

  @overload
  @abc.abstractmethod
  def query(self, prompt: str, *, return_response: t.Literal['raw'], **attrs: t.Any) -> DictStrAny:
    ...

  @overload
  @abc.abstractmethod
  def query(self, prompt: str, *, return_response: t.Literal['attrs'], **attrs: t.Any) -> openllm_core.GenerationOutput:
    ...

  @abc.abstractmethod
  def query(self, prompt: str, return_response: t.Literal['attrs', 'raw', 'processed'] = 'processed', **attrs: t.Any) -> t.Any:
    raise NotImplementedError

  # NOTE: Scikit interface
  @overload
  @abc.abstractmethod
  def predict(self, prompt: str, *, return_response: t.Literal['processed'], **attrs: t.Any) -> str:
    ...

  @overload
  @abc.abstractmethod
  def predict(self, prompt: str, *, return_response: t.Literal['raw'], **attrs: t.Any) -> DictStrAny:
    ...

  @overload
  @abc.abstractmethod
  def predict(self, prompt: str, *, return_response: t.Literal['attrs'], **attrs: t.Any) -> openllm_core.GenerationOutput:
    ...

  @abc.abstractmethod
  def predict(self, prompt: str, **attrs: t.Any) -> t.Any:
    raise NotImplementedError

  @functools.cached_property
  def _hf_agent(self) -> transformers.HfAgent:
    if not is_transformers_available():
      raise RuntimeError("transformers is required to use HF agent. Install with 'pip install \"openllm-client[agents]\"'.")
    if not self.supports_hf_agent:
      raise RuntimeError(f'{self.model_name} ({self.backend}) does not support running HF agent.')
    import transformers
    return transformers.HfAgent(urljoin(self._address, '/hf/agent'))

  @property
  def _metadata(self) -> t.Any:
    return self.call('metadata')

  @property
  def model_name(self) -> str:
    try:
      return self._metadata['model_name']
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def model_id(self) -> str:
    try:
      return self._metadata['model_id']
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def backend(self) -> LiteralBackend:
    try:
      return self._metadata['backend']
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def timeout(self) -> int:
    try:
      return self._metadata['timeout']
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def configuration(self) -> dict[str, t.Any]:
    try:
      return orjson.loads(self._metadata['configuration'])
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def supports_embeddings(self) -> bool:
    try:
      return self._metadata.get('supports_embeddings', False)
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def supports_hf_agent(self) -> bool:
    try:
      return self._metadata.get('supports_hf_agent', False)
    except KeyError:
      raise RuntimeError('Malformed service endpoint. (Possible malicious)') from None

  @property
  def config(self) -> openllm_core.LLMConfig:
    return openllm_core.AutoConfig.for_model(self.model_name).model_construct_env(**self.configuration)

  @functools.cached_property
  def inner(self) -> t.Any:
    raise NotImplementedError("'inner' client is not implemented.")

class _Client(_ClientAttr):
  _host: str
  _port: str

  def call(self, api_name: str, *args: t.Any, **attrs: t.Any) -> t.Any:
    return self.inner.call(f'{api_name}_{self._api_version}', *args, **attrs)

  def health(self) -> t.Any:
    return self.inner.health()

  @functools.cached_property
  def inner(self) -> BentoClient:
    BentoClient.wait_until_server_ready(self._host, int(self._port), timeout=self._timeout)
    return BentoClient.from_url(self._address)

  # Agent integration
  def ask_agent(self, task: str, *, return_code: bool = False, remote: bool = False, agent_type: LiteralString = 'hf', **attrs: t.Any) -> t.Any:
    if agent_type == 'hf': return self._run_hf_agent(task, return_code=return_code, remote=remote, **attrs)
    else: raise RuntimeError(f"Unknown 'agent_type={agent_type}'")

  def _run_hf_agent(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    if len(args) > 1: raise ValueError("'args' should only take one positional argument.")
    task = kwargs.pop('task', args[0])
    return_code = kwargs.pop('return_code', False)
    remote = kwargs.pop('remote', False)
    try:
      return self._hf_agent.run(task, return_code=return_code, remote=remote, **kwargs)
    except Exception as err:
      logger.error('Exception caught while sending instruction to HF agent: %s', err, exc_info=err)
      logger.info("Tip: LLMServer at '%s' might not support 'generate_one'.", self._address)

class _AsyncClient(_ClientAttr):
  _host: str
  _port: str

  def __init__(self, address: str, timeout: float = 30):
    self._address, self._timeout = address, timeout

  async def call(self, api_name: str, *args: t.Any, **attrs: t.Any) -> t.Any:
    return await self.inner.call(f'{api_name}_{self._api_version}', *args, **attrs)

  async def health(self) -> t.Any:
    return await self.inner.health()

  @functools.cached_property
  def inner(self) -> AsyncBentoClient:
    ensure_exec_coro(AsyncBentoClient.wait_until_server_ready(self._host, int(self._port), timeout=self._timeout))
    return ensure_exec_coro(AsyncBentoClient.from_url(self._address))

  # Agent integration
  async def ask_agent(self, task: str, *, return_code: bool = False, remote: bool = False, agent_type: LiteralString = 'hf', **attrs: t.Any) -> t.Any:
    if agent_type == 'hf': return await self._run_hf_agent(task, return_code=return_code, remote=remote, **attrs)
    else: raise RuntimeError(f"Unknown 'agent_type={agent_type}'")

  async def _run_hf_agent(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
    if len(args) > 1: raise ValueError("'args' should only take one positional argument.")
    from transformers.tools.agents import clean_code_for_run
    from transformers.tools.agents import get_tool_creation_code
    from transformers.tools.agents import resolve_tools
    from transformers.tools.python_interpreter import evaluate

    task = kwargs.pop('task', args[0])
    return_code = kwargs.pop('return_code', False)
    remote = kwargs.pop('remote', False)
    stop = ['Task:']
    prompt = t.cast(str, self._hf_agent.format_prompt(task))
    async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
      response = await client.post(self._hf_agent.url_endpoint, json={'inputs': prompt, 'parameters': {'max_new_tokens': 200, 'return_full_text': False, 'stop': stop}})
      if response.status_code != HTTPStatus.OK: raise ValueError(f'Error {response.status_code}: {response.json()}')

    result = response.json()[0]['generated_text']
    # Inference API returns the stop sequence
    for stop_seq in stop:
      if result.endswith(stop_seq):
        result = result[:-len(stop_seq)]
        break
    # the below have the same logic as agent.run API
    explanation, code = clean_code_for_run(result)
    self._hf_agent.log(f'==Explanation from the agent==\n{explanation}')
    self._hf_agent.log(f'\n\n==Code generated by the agent==\n{code}')
    if not return_code:
      self._hf_agent.log('\n\n==Result==')
      self._hf_agent.cached_tools = resolve_tools(code, self._hf_agent.toolbox, remote=remote, cached_tools=self._hf_agent.cached_tools)
      return evaluate(code, self._hf_agent.cached_tools, state=kwargs.copy())
    else:
      tool_code = get_tool_creation_code(code, self._hf_agent.toolbox, remote=remote)
      return f'{tool_code}\n{code}'

class BaseClient(_Client):
  def chat(self, prompt: str, history: list[str], **attrs: t.Any) -> str:
    raise NotImplementedError

  def embed(self, prompt: t.Sequence[str] | str) -> openllm_core.EmbeddingsOutput:
    return openllm_core.EmbeddingsOutput(**self.call('embeddings', list([prompt] if isinstance(prompt, str) else prompt)))

  def predict(self, prompt: str, **attrs: t.Any) -> openllm_core.GenerationOutput | DictStrAny | str:
    return self.query(prompt, **attrs)

  def query(self, prompt: str, return_response: t.Literal['attrs', 'raw', 'processed'] = 'processed', **attrs: t.Any) -> t.Any:
    return_raw_response = attrs.pop('return_raw_response', None)
    if return_raw_response is not None:
      logger.warning("'return_raw_response' is now deprecated. Please use 'return_response=\"raw\"' instead.")
      if return_raw_response is True: return_response = 'raw'
    return_attrs = attrs.pop('return_attrs', None)
    if return_attrs is not None:
      logger.warning("'return_attrs' is now deprecated. Please use 'return_response=\"attrs\"' instead.")
      if return_attrs is True: return_response = 'attrs'
    use_default_prompt_template = attrs.pop('use_default_prompt_template', False)
    prompt, generate_kwargs, postprocess_kwargs = self.config.sanitize_parameters(prompt, use_default_prompt_template=use_default_prompt_template, **attrs)
    r = openllm_core.GenerationOutput(**self.call('generate', openllm_core.GenerationInput(prompt=prompt, llm_config=self.config.model_construct_env(**generate_kwargs)).model_dump()))
    if return_response == 'attrs': return r
    elif return_response == 'raw': return bentoml_cattr.unstructure(r)
    else: return self.config.postprocess_generate(prompt, r.responses, **postprocess_kwargs)

class BaseAsyncClient(_AsyncClient):
  async def chat(self, prompt: str, history: list[str], **attrs: t.Any) -> str:
    raise NotImplementedError

  async def embed(self, prompt: t.Sequence[str] | str) -> openllm_core.EmbeddingsOutput:
    return openllm_core.EmbeddingsOutput(**(await self.call('embeddings', list([prompt] if isinstance(prompt, str) else prompt))))

  async def predict(self, prompt: str, **attrs: t.Any) -> t.Any:
    return await self.query(prompt, **attrs)

  async def query(self, prompt: str, return_response: t.Literal['attrs', 'raw', 'processed'] = 'processed', **attrs: t.Any) -> t.Any:
    return_raw_response = attrs.pop('return_raw_response', None)
    if return_raw_response is not None:
      logger.warning("'return_raw_response' is now deprecated. Please use 'return_response=\"raw\"' instead.")
      if return_raw_response is True: return_response = 'raw'
    return_attrs = attrs.pop('return_attrs', None)
    if return_attrs is not None:
      logger.warning("'return_attrs' is now deprecated. Please use 'return_response=\"attrs\"' instead.")
      if return_attrs is True: return_response = 'attrs'
    use_default_prompt_template = attrs.pop('use_default_prompt_template', False)
    prompt, generate_kwargs, postprocess_kwargs = self.config.sanitize_parameters(prompt, use_default_prompt_template=use_default_prompt_template, **attrs)
    r = openllm_core.GenerationOutput(**(await self.call('generate', openllm_core.GenerationInput(prompt=prompt, llm_config=self.config.model_construct_env(**generate_kwargs)).model_dump())))
    if return_response == 'attrs': return r
    elif return_response == 'raw': return bentoml_cattr.unstructure(r)
    else: return self.config.postprocess_generate(prompt, r.responses, **postprocess_kwargs)
