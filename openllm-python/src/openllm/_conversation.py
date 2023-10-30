from __future__ import annotations
import typing as t

from enum import IntEnum
from enum import auto

import attr

if t.TYPE_CHECKING: import openllm_core

_object_setattr = object.__setattr__

class SeparatorStyle(IntEnum):
  '''Separator styles.'''

  # Generic separator styles for chat models
  ADD_COLON_SINGLE = auto()
  ADD_COLON_TWO = auto()
  ADD_COLON_SPACE_SINGLE = auto()
  NO_COLON_SINGLE = auto()
  NO_COLON_TWO = auto()
  ADD_NEW_LINE_SINGLE = auto()

  # Special separator styles for specific chat models in OpenLLM
  LLAMA = auto()
  CHATGLM = auto()
  DOLLY = auto()
  MPT = auto()
  STARCODER = auto()

@attr.define
class Conversation:
  '''A class that manages prompt templates and keeps all conversation history.'''

  # The name of this template
  name: str
  # The template of the system prompt
  system_template: str = '{system_message}'
  # The system message
  system_message: str = ''
  # The names of two roles
  roles: t.Tuple[str, str] = ('User', 'Assistant')
  # All messages. Each item is (role, message).
  messages: t.List[t.List[str]] = []
  # The number of few shot examples
  offset: int = 0
  # The separator style and configurations
  sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
  sep: str = '\n'
  sep2: str = ''
  # Stop criteria (the default one is EOS token)
  stop_str: t.Union[str, t.List[str]] = ''
  # Stops generation if meeting any token in this list
  stop_token_ids: t.List[int] = []

  def get_prompt(self) -> str:
    '''Get the prompt for generation.'''
    system_prompt = self.system_template.format(system_message=self.system_message)

    # Generic separator styles for chat models
    if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:  # Role with colon
      ret = system_prompt + self.sep
      for role, message in self.messages:
        if message:
          ret += role + ': ' + message + self.sep
        else:
          ret += role + ':'
      return ret
    elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:  # Role with colon, two different separators for two roles
      seps = [self.sep, self.sep2]
      ret = system_prompt + seps[0]
      for i, (role, message) in enumerate(self.messages):
        if message:
          ret += role + ': ' + message + seps[i % 2]
        else:
          ret += role + ':'
      return ret
    elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:  # Add a space after colon
      ret = system_prompt + self.sep
      for role, message in self.messages:
        if message:
          ret += role + ': ' + message + self.sep
        else:
          ret += role + ': '  # must be end with a space
      return ret
    elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:  # Add a new line after role
      ret = '' if system_prompt == '' else system_prompt + self.sep
      for role, message in self.messages:
        if message:
          ret += role + '\n' + message + self.sep
        else:
          ret += role + '\n'
      return ret
    elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:  # No colon
      ret = system_prompt
      for role, message in self.messages:
        if message:
          ret += role + message + self.sep
        else:
          ret += role
      return ret
    elif self.sep_style == SeparatorStyle.NO_COLON_TWO:  # No colon, two different separators for two roles
      seps = [self.sep, self.sep2]
      ret = system_prompt
      for i, (role, message) in enumerate(self.messages):
        if message:
          ret += role + message + seps[i % 2]
        else:
          ret += role
      return ret
    # Special separator styles for specific chat models
    elif self.sep_style == SeparatorStyle.LLAMA:
      seps = [self.sep, self.sep2]
      if self.system_message:
        ret = system_prompt
      else:
        ret = '[INST] '
      for i, (role, message) in enumerate(self.messages):
        tag = self.roles[i % 2]
        if message:
          if i == 0:
            ret += message + ' '
          else:
            ret += tag + ' ' + message + seps[i % 2]
        else:
          ret += tag
      return ret
    elif self.sep_style == SeparatorStyle.CHATGLM:
      round_add_n = 1 if self.name == 'chatglm2' else 0
      if system_prompt:
        ret = system_prompt + self.sep
      else:
        ret = ''
      for i, (role, message) in enumerate(self.messages):
        if i % 2 == 0:
          ret += f'[Round {i//2 + round_add_n}]{self.sep}'
        if message:
          ret += f'{role}:{message}{self.sep}'
        else:
          ret += f'{role}:'
      return ret
    elif self.sep_style == SeparatorStyle.DOLLY:
      seps = [self.sep, self.sep2]
      ret = system_prompt
      for i, (role, message) in enumerate(self.messages):
        if message:
          ret += role + ':\n' + message + seps[i % 2]
          if i % 2 == 1:
            ret += '\n\n'
        else:
          ret += role + ':\n'
      return ret
    elif self.sep_style == SeparatorStyle.MPT:
      if system_prompt:
        ret = f'<|im_start|>system\n{system_prompt}<|im_end|>{self.sep}'
      else:
        ret = ''
      for i, (role, message) in enumerate(self.messages):
        if message:
          ret += f'<|im_start|>{role}\n{message}<|im_end|>{self.sep}'
        else:
          ret += f'{role}:'
      return ret
    elif self.sep_style == SeparatorStyle.STARCODER:
      if system_prompt:
        ret = f'<|system|>\n{system_prompt}<|end|>{self.sep}'
      else:
        ret = ''
      for i, (role, message) in enumerate(self.messages):
        if message:
          ret += f'{role}\n{message}<|end|>{self.sep}'
        else:
          ret += f'{role}:'
    else:
      raise ValueError(f'Invalid style: {self.sep_style}')
    return ret

  def set_system_message(self, system_message: str) -> None: _object_setattr(self, 'system_message', system_message)
  def append_message(self, role: str, message: str) -> None:
    '''Append a new message.'''
    self.messages.append([role, message])

  def update_last_message(self, message: str) -> None:
    '''Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        '''
    self.messages[-1][1] = message

  def to_openai_api_messages(self) -> t.List[t.Dict[str, str]]:
    '''Convert the conversation to OpenAI chat completion format.'''
    ret = [{'role': 'system', 'content': self.system_message}]

    for i, (_, msg) in enumerate(self.messages[self.offset:]):
      if i % 2 == 0:
        ret.append({'role': 'user', 'content': msg})
      elif msg is not None:
        ret.append({'role': 'assistant', 'content': msg})
    return ret

# A global registry for all conversation templates for OpenLLM models
conv_templates: t.Dict[str, Conversation] = {}

def register_conv_template(template: Conversation) -> None:
  '''Register a new conversation template.'''
  conv_templates[template.name] = template

def get_conv_template(name: str, llm_config: openllm_core.LLMConfig) -> Conversation:
  if name not in conv_templates: raise ValueError(f"Failed to find conversation templates for {name}")
  template = conv_templates[name]
  if hasattr(llm_config, 'default_system_message'): template.set_system_message(llm_config.default_system_message)
  return template

# Raw template
register_conv_template(Conversation(name='raw', system_message='', roles=('', ''), sep_style=SeparatorStyle.NO_COLON_SINGLE, sep=''))

# Llama template
# source: https://huggingface.co/blog/codellama#conversational-instructions
register_conv_template(Conversation(name='llama', system_template='[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n', roles=('[INST]', '[/INST]'), sep_style=SeparatorStyle.LLAMA, sep=' ', sep2=' </s><s>',))

# ChatGLM template
register_conv_template(Conversation(name='chatglm', roles=('问', '答'), sep_style=SeparatorStyle.CHATGLM, sep='\n',))

# Dolly-v2 template
register_conv_template(
    Conversation(name='dolly_v2',
                 system_message='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n',
                 roles=('### Instruction', '### Response'),
                 sep_style=SeparatorStyle.DOLLY,
                 sep='\n\n',
                 sep2='### End',
                 ))

# Falcon template
register_conv_template(
    # source: https://huggingface.co/tiiuae/falcon-7b-instruct/discussions/1
    Conversation(name='falcon', roles=('User', 'Assistant'), messages=[], sep_style=SeparatorStyle.ADD_COLON_SINGLE,  #  No space after colon
                 sep='\n',
                 ))

# Flan-T5 default template
register_conv_template(
    # source: https://www.philschmid.de/fine-tune-flan-t5
    # No specific template found, but seems to have the same dialogue style
    Conversation(name='flan-t5', system_message='', roles=('User', 'Assistant'), sep_style=SeparatorStyle.ADD_COLON_SINGLE, sep='\n'))

# GPT-NeoX default template
register_conv_template(
    # source: https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B
    # Don't know if GPT-NeoX-20B is trained on any chat prompt template
    Conversation(name='gpt-neox', system_message='', roles=('<human>', '<bot>'), sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE, sep='\n'))

# MPT template
register_conv_template(
    # source: https://huggingface.co/TheBloke/mpt-30B-chat-GGML/discussions/4
    Conversation(name='mpt', roles=('user', 'assistant'), messages=[], sep_style=SeparatorStyle.MPT, sep='\n'))

# OPT template (No reference for OPT found)
register_conv_template(Conversation(name='opt', roles=('User', 'Assistant'), messages=[], sep_style=SeparatorStyle.ADD_COLON_SINGLE, sep='\n'))

# StableLM default template
register_conv_template(
    Conversation(name='stablelm',
                 system_template='<|SYSTEM|>{system_message}',
                 system_message='''# StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
''',
                 roles=('<|USER|>', '<|ASSISTANT|>'),
                 sep_style=SeparatorStyle.NO_COLON_SINGLE,
                 sep='',
                 stop_token_ids=[50278, 50279, 50277, 1, 0],
                 ))

# StarCoder default template
register_conv_template(
    # source: https://github.com/bigcode-project/starcoder/blob/main/chat/dialogues.py
    Conversation(name='starcoder', system_message='', roles=('<|user|>', '<|assistant|>'), sep_style=SeparatorStyle.STARCODER, sep='\n'))

# Baichuan default template
register_conv_template(
    # source: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/19ef51ba5bad8935b03acd20ff04a269210983bc/modeling_baichuan.py#L555
    # https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/blob/main/generation_config.json
    # https://github.com/baichuan-inc/Baichuan-13B/issues/25
    Conversation(name='baichuan', roles=('<reserved_102>', '<reserved_103>'), sep_style=SeparatorStyle.NO_COLON_SINGLE, sep=''))

# Mistral template
register_conv_template(Conversation(name='mistral', system_message='', roles=('[INST]', '[/INST]'), sep_style=SeparatorStyle.LLAMA, sep=' ', sep2='</s>',))
