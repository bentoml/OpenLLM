from __future__ import annotations
import logging
import string
import typing as t

import attr
import click
import inflection
import orjson
from bentoml_cli.utils import opt_callback

import openllm
from openllm_cli import termui
from openllm_cli._factory import model_complete_envvar

logger = logging.getLogger(__name__)


class PromptFormatter(string.Formatter):
  def vformat(self, format_string: str, args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]) -> t.Any:
    if len(args) > 0:
      raise ValueError('Positional arguments are not supported')
    return super().vformat(format_string, args, kwargs)

  def check_unused_args(self, used_args: set[int | str], args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]) -> None:
    extras = set(kwargs).difference(used_args)
    if extras:
      raise KeyError(f'Extra params passed: {extras}')

  def extract_template_variables(self, template: str) -> t.Sequence[str]:
    return [field[1] for field in self.parse(template) if field[1] is not None]


default_formatter = PromptFormatter()

# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__


@attr.define(slots=True)
class PromptTemplate:
  template: str
  _input_variables: t.Sequence[str] = attr.field(init=False)

  def __attrs_post_init__(self) -> None:
    self._input_variables = default_formatter.extract_template_variables(self.template)

  def with_options(self, **attrs: t.Any) -> PromptTemplate:
    prompt_variables = {key: '{' + key + '}' if key not in attrs else attrs[key] for key in self._input_variables}
    o = attr.evolve(self, template=self.template.format(**prompt_variables))
    _object_setattr(o, '_input_variables', default_formatter.extract_template_variables(o.template))
    return o

  def format(self, **attrs: t.Any) -> str:
    prompt_variables = {k: v for k, v in attrs.items() if k in self._input_variables}
    try:
      return self.template.format(**prompt_variables)
    except KeyError as e:
      raise RuntimeError(f"Missing variable '{e.args[0]}' (required: {self._input_variables}) in the prompt template.") from None


@click.command('get_prompt', context_settings=termui.CONTEXT_SETTINGS)
@click.argument('model_id', shell_complete=model_complete_envvar)
@click.argument('prompt', type=click.STRING)
@click.option('--prompt-template-file', type=click.File(), default=None)
@click.option('--chat-template-file', type=click.File(), default=None)
@click.option('--system-message', type=str, default=None)
@click.option(
  '--add-generation-prompt/--no-add-generation-prompt',
  default=False,
  help='See https://huggingface.co/docs/transformers/main/chat_templating#what-template-should-i-use. This only applicable if model-id is a HF model_id',
)
@click.option(
  '--opt',
  help="Define additional prompt variables. (format: ``--opt system_message='You are a useful assistant'``)",
  required=False,
  multiple=True,
  callback=opt_callback,
  metavar='ARG=VALUE[,ARG=VALUE]',
)
@click.pass_context
def cli(
  ctx: click.Context,
  /,
  model_id: str,
  prompt: str,
  prompt_template_file: t.IO[t.Any] | None,
  chat_template_file: t.IO[t.Any] | None,
  system_message: str | None,
  add_generation_prompt: bool,
  _memoized: dict[str, t.Any],
  **_: t.Any,
) -> str | None:
  """Helpers for generating prompts.

  \b
  It accepts remote HF model_ids as well as model name passed to `openllm start`.

  If you pass in a HF model_id, then it will use the tokenizer to generate the prompt.

  ```bash
  openllm get-prompt WizardLM/WizardCoder-15B-V1.0 "Hello there"
  ```

  If you need change the prompt template, you can create the template file that contains the jina2 template through `--chat-template-file`
  See https://huggingface.co/docs/transformers/main/chat_templating#templates-for-chat-models for more details.

  \b
  ```bash
  openllm get-prompt WizardLM/WizardCoder-15B-V1.0 "Hello there" --chat-template-file template.jinja2
  ```

  \b

  If you pass a model name, then it will use OpenLLM configuration to generate the prompt.
  Note that this is mainly for utilities, as OpenLLM won't use these prompts to format for you.

  \b
  ```bash
  openllm get-prompt mistral "Hello there"
  """
  _memoized = {k: v[0] for k, v in _memoized.items() if v}

  if prompt_template_file and chat_template_file:
    ctx.fail('prompt-template-file and chat-template-file are mutually exclusive.')

  acceptable = set(openllm.CONFIG_MAPPING_NAMES.keys()) | set(inflection.dasherize(name) for name in openllm.CONFIG_MAPPING_NAMES.keys())
  if model_id in acceptable:
    logger.warning('Using a default prompt from OpenLLM. Note that this prompt might not work for your intended usage.\n')
    config = openllm.AutoConfig.for_model(model_id)
    template = prompt_template_file.read() if prompt_template_file is not None else config.template
    system_message = system_message or config.system_message

    try:
      formatted = PromptTemplate(template).with_options(system_message=system_message).format(instruction=prompt, **_memoized)
    except RuntimeError as err:
      logger.debug('Exception caught while formatting prompt: %s', err)
      ctx.fail(str(err))
  else:
    import transformers

    trust_remote_code = openllm.utils.check_bool_env('TRUST_REMOTE_CODE', False)
    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if chat_template_file is not None:
      chat_template_file = chat_template_file.read()
    if system_message is None:
      logger.warning('system-message is not provided, using default infer from the model architecture.\n')
      for architecture in config.architectures:
        if architecture in openllm.AutoConfig._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE():
          system_message = (
            openllm.AutoConfig.infer_class_from_name(openllm.AutoConfig._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE()[architecture])
            .model_construct_env()
            .system_message
          )
          break
      else:
        ctx.fail(f'Failed to infer system message from model architecture: {config.architectures}. Please pass in --system-message')
    messages = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': prompt}]
    formatted = tokenizer.apply_chat_template(messages, chat_template=chat_template_file, add_generation_prompt=add_generation_prompt, tokenize=False)

  termui.echo(orjson.dumps({'prompt': formatted}, option=orjson.OPT_INDENT_2).decode(), fg='white')
  ctx.exit(0)
