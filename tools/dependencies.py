#!/usr/bin/env python3
from __future__ import annotations
import dataclasses
import os
import sys
import typing as t

import inflection
import tomlkit
from ghapi.all import GhApi

if t.TYPE_CHECKING:
  from tomlkit.items import Array, Table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'openllm-python', 'src'))
sys.path.insert(1, os.path.join(ROOT, 'openllm-core', 'src'))

import openllm

_OWNER, _REPO = 'bentoml', 'openllm'


@dataclasses.dataclass(frozen=True)
class Classifier:
  identifier: t.Dict[str, str] = dataclasses.field(
    default_factory=lambda: {
      'status': 'Development Status',
      'environment': 'Environment',
      'license': 'License',
      'topic': 'Topic',
      'os': 'Operating System',
      'audience': 'Intended Audience',
      'typing': 'Typing',
      'language': 'Programming Language',
    }
  )
  joiner: str = ' :: '

  @staticmethod
  def status() -> dict[int, str]:
    return {
      v: status
      for v, status in zip(
        range(1, 8),
        [
          '1 - Planning',
          '2 - Pre-Alpha',
          '3 - Alpha',
          '4 - Beta',
          '5 - Production/Stable',
          '6 - Mature',
          '7 - Inactive',
        ],
      )
    }

  @staticmethod
  def apache() -> str:
    return Classifier.create_classifier('license', 'OSI Approved', 'Apache Software License')

  @staticmethod
  def create_classifier(identifier: str, *decls: t.Any) -> str:
    cls_ = Classifier()
    if identifier not in cls_.identifier:
      raise ValueError(f'{identifier} is not yet supported (supported alias: {Classifier.identifier})')
    return cls_.joiner.join([cls_.identifier[identifier], *decls])

  @staticmethod
  def create_python_classifier(
    implementation: list[str] | None = None, supported_version: list[str] | None = None
  ) -> list[str]:
    if supported_version is None:
      supported_version = ['3.8', '3.9', '3.10', '3.11', '3.12']
    if implementation is None:
      implementation = ['CPython', 'PyPy']
    base = [
      Classifier.create_classifier('language', 'Python'),
      Classifier.create_classifier('language', 'Python', '3'),
    ]
    base.append(Classifier.create_classifier('language', 'Python', '3', 'Only'))
    base.extend([Classifier.create_classifier('language', 'Python', version) for version in supported_version])
    base.extend([
      Classifier.create_classifier('language', 'Python', 'Implementation', impl) for impl in implementation
    ])
    return base

  @staticmethod
  def create_status_classifier(level: int) -> str:
    return Classifier.create_classifier('status', Classifier.status()[level])


@dataclasses.dataclass(frozen=True)
class Dependencies:
  name: str
  git_repo_url: t.Optional[str] = None
  branch: t.Optional[str] = None
  extensions: t.Optional[t.List[str]] = None
  subdirectory: t.Optional[str] = None
  requires_gpu: bool = False
  lower_constraint: t.Optional[str] = None
  upper_constraint: t.Optional[str] = None
  platform: t.Optional[t.Tuple[t.Literal['Linux', 'Windows', 'Darwin'], t.Literal['eq', 'ne']]] = None

  def with_options(self, **kwargs: t.Any) -> Dependencies:
    return dataclasses.replace(self, **kwargs)

  @property
  def has_constraint(self) -> bool:
    return self.lower_constraint is not None or self.upper_constraint is not None

  @property
  def pypi_extensions(self) -> str:
    return '' if self.extensions is None else f"[{','.join(self.extensions)}]"

  @staticmethod
  def platform_restriction(platform: t.LiteralString, op: t.Literal['eq', 'ne'] = 'eq') -> str:
    return f'platform_system{"==" if op == "eq" else "!="}"{platform}"'

  def to_str(self) -> str:
    deps: list[str] = []
    if self.lower_constraint is not None and self.upper_constraint is not None:
      dep = f'{self.name}{self.pypi_extensions}>={self.lower_constraint},<{self.upper_constraint}'
    elif self.lower_constraint is not None:
      dep = f'{self.name}{self.pypi_extensions}>={self.lower_constraint}'
    elif self.upper_constraint is not None:
      dep = f'{self.name}{self.pypi_extensions}<{self.upper_constraint}'
    elif self.subdirectory is not None:
      dep = f'{self.name}{self.pypi_extensions} @ git+https://github.com/{self.git_repo_url}.git#subdirectory={self.subdirectory}'
    elif self.branch is not None:
      dep = f'{self.name}{self.pypi_extensions} @ git+https://github.com/{self.git_repo_url}.git@{self.branch}'
    else:
      dep = f'{self.name}{self.pypi_extensions}'
    deps.append(dep)
    if self.platform:
      deps.append(self.platform_restriction(*self.platform))
    return ';'.join(deps)

  @classmethod
  def from_tuple(cls, *decls: t.Any) -> Dependencies:
    return cls(*decls)


_LOWER_BENTOML_CONSTRAINT = '1.2'
_BENTOML_EXT = ['io']
_TRANSFORMERS_EXT = ['torch', 'tokenizers']
_TRANSFORMERS_CONSTRAINTS = '4.36.0'

FINE_TUNE_DEPS = ['peft>=0.6.0', 'datasets', 'trl', 'huggingface-hub']
GRPC_DEPS = [f'bentoml[grpc]>={_LOWER_BENTOML_CONSTRAINT}']
OPENAI_DEPS = ['openai[datalib]>=1', 'tiktoken', 'fastapi']
AGENTS_DEPS = [f'transformers[agents]>={_TRANSFORMERS_CONSTRAINTS}', 'diffusers', 'soundfile']
PLAYGROUND_DEPS = ['jupyter', 'notebook', 'ipython', 'jupytext', 'nbformat']
GGML_DEPS = ['ctransformers']
AWQ_DEPS = ['autoawq']
GPTQ_DEPS = ['auto-gptq[triton]>=0.4.2']
VLLM_DEPS = ['vllm==0.4.2']

_base_requirements: dict[str, t.Any] = {
  inflection.dasherize(name): config_cls()['requirements']
  for name, config_cls in openllm.CONFIG_MAPPING.items()
  if 'requirements' in config_cls()
}

# shallow copy from locals()
_locals = locals().copy()

# NOTE: update this table when adding new external dependencies
# sync with openllm.utils.OPTIONAL_DEPENDENCIES
_base_requirements.update({
  v: _locals.get(f'{inflection.underscore(v).upper()}_DEPS') for v in openllm.utils.OPTIONAL_DEPENDENCIES
})

_base_requirements = {k: v for k, v in sorted(_base_requirements.items())}

fname = f'{os.path.basename(os.path.dirname(__file__))}/{os.path.basename(__file__)}'


def correct_style(it: t.Any) -> t.Any:
  return it


def create_classifiers() -> Array:
  arr = correct_style(tomlkit.array())
  arr.extend([
    Classifier.create_status_classifier(5),
    Classifier.create_classifier('environment', 'GPU', 'NVIDIA CUDA'),
    Classifier.create_classifier('environment', 'GPU', 'NVIDIA CUDA', '12'),
    Classifier.create_classifier('environment', 'GPU', 'NVIDIA CUDA', '11.8'),
    Classifier.create_classifier('environment', 'GPU', 'NVIDIA CUDA', '11.7'),
    Classifier.apache(),
    Classifier.create_classifier('topic', 'Scientific/Engineering', 'Artificial Intelligence'),
    Classifier.create_classifier('topic', 'Software Development', 'Libraries'),
    Classifier.create_classifier('os', 'OS Independent'),
    Classifier.create_classifier('audience', 'Developers'),
    Classifier.create_classifier('audience', 'Science/Research'),
    Classifier.create_classifier('audience', 'System Administrators'),
    Classifier.create_classifier('typing', 'Typed'),
    *Classifier.create_python_classifier(),
  ])
  return arr.multiline(True)


def create_optional_table() -> Table:
  all_array = tomlkit.array()
  all_array.append(f"openllm[{','.join([k for k, v in _base_requirements.items() if v])}]")

  table = tomlkit.table(is_super_table=True)
  _base_requirements.update({
    'full': correct_style(all_array.multiline(True)),
    'all': tomlkit.array('["openllm[full]"]'),
  })
  table.update({k: v for k, v in sorted(_base_requirements.items()) if v})
  table.add(tomlkit.nl())

  return table


def create_url_table(_info: t.Any) -> Table:
  table = tomlkit.table()
  _urls = {
    'Blog': 'https://modelserving.com',
    'Chat': 'https://discord.gg/openllm',
    'Documentation': 'https://github.com/bentoml/openllm#readme',
    'GitHub': _info.html_url,
    'History': f'{_info.html_url}/blob/main/CHANGELOG.md',
    'Homepage': _info.homepage,
    'Tracker': f'{_info.html_url}/issues',
    'Twitter': 'https://twitter.com/bentomlai',
  }
  table.update({k: v for k, v in sorted(_urls.items())})
  return table


def build_system() -> Table:
  table = tomlkit.table()
  table.add('build-backend', 'hatchling.build')
  requires_array = correct_style(tomlkit.array())
  requires_array.extend(['hatchling==1.18.0', 'hatch-vcs==0.3.0', 'hatch-fancy-pypi-readme==23.1.0'])
  table.add('requires', requires_array.multiline(True))
  return table


def keywords() -> Array:
  arr = correct_style(tomlkit.array())
  arr.extend([
    'MLOps',
    'AI',
    'BentoML',
    'Model Serving',
    'Model Deployment',
    'LLMOps',
    'Falcon',
    'Vicuna',
    'Llama 2',
    'Fine tuning',
    'Serverless',
    'Large Language Model',
    'Generative AI',
    'StableLM',
    'Alpaca',
    'PyTorch',
    'Mistral',
    'vLLM',
    'Transformers',
  ])
  return arr.multiline(True)


def build_cli_extensions() -> Table:
  table = tomlkit.table()
  table.update({'openllm': '_openllm_tiny._entrypoint:cli'})
  return table


def main(args) -> int:
  api = GhApi(owner=_OWNER, repo=_REPO, authenticate=False)
  _info = api.repos.get()
  with open(os.path.join(ROOT, 'openllm-python', 'pyproject.toml'), 'r') as f:
    pyproject = tomlkit.parse(f.read())

  if args.release_version is not None:
    release_version = args.release_version
  else:
    release_version = openllm.bundle.RefResolver.from_strategy('release').version

  _BASE_DEPENDENCIES = [
    Dependencies(name='bentoml', extensions=_BENTOML_EXT, lower_constraint=_LOWER_BENTOML_CONSTRAINT),
    Dependencies(name='transformers', extensions=_TRANSFORMERS_EXT, lower_constraint=_TRANSFORMERS_CONSTRAINTS),
    Dependencies(name='openllm-client', lower_constraint=release_version),
    Dependencies(name='openllm-core', lower_constraint=release_version),
    Dependencies(name='safetensors'),
    Dependencies(name='vllm', lower_constraint='0.4.2'),
    Dependencies(name='optimum', lower_constraint='1.12.0'),
    Dependencies(name='accelerate'),
    Dependencies(name='ghapi'),
    Dependencies(name='einops'),
    Dependencies(name='sentencepiece'),
    Dependencies(name='scipy'),
    Dependencies(name='build', upper_constraint='1', extensions=['virtualenv']),
    Dependencies(name='click', lower_constraint='8.1.3'),
    Dependencies(name='cuda-python', platform=('Darwin', 'ne')),
    Dependencies(name='bitsandbytes', upper_constraint='0.42'),  # 0.41 works with CUDA 11.8
  ]

  dependencies_array = correct_style(tomlkit.array())
  dependencies_array.extend([v.to_str() for v in _BASE_DEPENDENCIES])
  # dynamic field
  dyn_arr = tomlkit.array()
  dyn_arr.extend(['version', 'readme'])

  pyproject['build-system'] = build_system()
  pyproject['project']['classifiers'] = create_classifiers()
  pyproject['project']['dependencies'] = dependencies_array.multiline(True)
  pyproject['project']['description'] = f'{_info.name}: {_info.description}'
  pyproject['project']['dynamic'] = dyn_arr
  pyproject['project']['keywords'] = keywords()
  pyproject['project']['license'] = _info.license.spdx_id
  pyproject['project']['name'] = f'{_info.name.lower()}'
  pyproject['project']['requires-python'] = '>=3.8'

  pyproject['project']['urls'] = create_url_table(_info)
  pyproject['project']['scripts'] = build_cli_extensions()
  pyproject['project']['optional-dependencies'] = create_optional_table()

  with open(os.path.join(ROOT, 'openllm-python', 'pyproject.toml'), 'w') as f:
    f.write(tomlkit.dumps(pyproject))
  return 0


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--release-version', type=str, default=None)
  raise SystemExit(main(parser.parse_args()))
