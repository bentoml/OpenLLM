from __future__ import annotations
import typing as t
from enum import IntEnum, auto

import attr

_object_setattr = object.__setattr__


class SeparatorStyle(IntEnum):
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
  # The name of this template
  name: str
  # The template of the system prompt
  system_template: str = '{system_message}'
  # The system message
  system_message: str = ''
  # The names of two roles
  roles: t.Tuple[str, str] = ('User', 'Assistant')
  # All messages. Each item is (role, message).
  messages: t.List[t.List[str]] = attr.field(factory=list)
  # The number of few shot examples
  offset: int = 0
  # The separator style and configurations
  sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
  sep: str = '\n'
  sep2: str = ''
  # Stop criteria (the default one is EOS token)
  stop_str: t.Union[str, t.List[str]] = ''
  # Stops generation if meeting any token in this list
  stop_token_ids: t.List[int] = attr.field(factory=list)

  def get_prompt(self) -> str:
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
      ret = system_prompt if self.system_message else '<s>[INST] '
      for i, (_, message) in enumerate(self.messages):
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
      ret = system_prompt + self.sep if system_prompt else ''
      for i, (role, message) in enumerate(self.messages):
        if i % 2 == 0:
          ret += f'[Round {i//2 + round_add_n}]{self.sep}'
        ret += f'{role}:{message}{self.sep}' if message else f'{role}:'
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
      ret = f'<|im_start|>system\n{system_prompt}<|im_end|>{self.sep}' if system_prompt else ''
      for _, (role, message) in enumerate(self.messages):
        ret += f'<|im_start|>{role}\n{message}<|im_end|>{self.sep}' if message else f'{role}:'
      return ret
    elif self.sep_style == SeparatorStyle.STARCODER:
      ret = f'<|system|>\n{system_prompt}<|end|>{self.sep}' if system_prompt else ''
      for _, (role, message) in enumerate(self.messages):
        ret += f'{role}\n{message}<|end|>{self.sep}' if message else f'{role}:'
    else:
      raise ValueError(f'Invalid style: {self.sep_style}')
    return ret

  # yapf: disable
  # The last message is typically set to be None when constructing the prompt,
  # so we need to update it in-place after getting the response from a model.
  def update_last_message(self,message: str)->None:self.messages[-1][1]=message
  def append_message(self,role: str,message: str)->None:self.messages.append([role,message])
  def set_system_message(self,system_message: str)->None:_object_setattr(self,'system_message',system_message)
  def with_options(self,**attrs:t.Any)->Conversation:return attr.evolve(self,**attrs)
  # yapf: enable

  def to_openai_messages(self) -> t.List[t.Dict[str, str]]:
    ret = [{'role': 'system', 'content': self.system_message}]
    for i, (_, msg) in enumerate(self.messages[self.offset :]):
      if i % 2 == 0:
        ret.append({'role': 'user', 'content': msg})
      elif msg is not None:
        ret.append({'role': 'assistant', 'content': msg})
    return ret

  def copy(self) -> Conversation:
    return Conversation(
      name=self.name,
      system_template=self.system_template,
      system_message=self.system_message,
      roles=self.roles,
      messages=self.messages,
      offset=self.offset,
      sep_style=self.sep_style,
      sep=self.sep,
      sep2=self.sep2,
      stop_str=self.stop_str,
      stop_token_ids=self.stop_token_ids,
    )
