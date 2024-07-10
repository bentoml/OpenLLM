import datetime
import re
import shutil

import pyaml
import questionary
import typer

from openllm.analytic import OpenLLMTyper
from openllm.common import INTERACTIVE, REPO_DIR, VERBOSE_LEVEL, RepoInfo, load_config, output, save_config

UPDATE_INTERVAL = datetime.timedelta(days=3)

app = OpenLLMTyper(help='manage repos')


@app.command()
def list(verbose: bool = False):
  if verbose:
    VERBOSE_LEVEL.set(20)
  config = load_config()
  pyaml.pprint([parse_repo_url(repo, name) for name, repo in config.repos.items()], sort_dicts=False, sort_keys=False)


@app.command()
def remove(name: str):
  config = load_config()
  if name not in config.repos:
    output(f'Repo {name} does not exist', style='red')
    return

  del config.repos[name]
  save_config(config)
  output(f'Repo {name} removed', style='green')


def _complete_alias(repo_name: str):
  from openllm.model import list_bento

  for bento in list_bento(repo_name=repo_name):
    alias = bento.labels.get('openllm_alias', '').strip()
    if alias:
      for a in alias.split(','):
        with open(bento.path.parent / a, 'w') as f:
          f.write(bento.version)


@app.command()
def update():
  import dulwich
  import dulwich.errors
  import dulwich.porcelain

  config = load_config()
  repos_in_use = set()
  for repo_name, repo in config.repos.items():
    repo = parse_repo_url(repo, repo_name)
    repos_in_use.add((repo.server, repo.owner, repo.repo))
    if repo.path.exists():  # TODO: use update instead of remove and clone
      shutil.rmtree(repo.path, ignore_errors=True)
    if not repo.path.exists():
      repo.path.parent.mkdir(parents=True, exist_ok=True)
      try:
        dulwich.porcelain.clone(
          f'https://{repo.server}/{repo.owner}/{repo.repo}.git',
          str(repo.path),
          checkout=True,
          depth=1,
          branch=repo.branch,
        )
        output('')
        output(f'Repo `{repo.name}` updated', style='green')
      except:
        shutil.rmtree(repo.path, ignore_errors=True)
        output(f'Failed to clone repo {repo.name}', style='red')
    else:
      try:
        import dulwich.porcelain

        dulwich.porcelain.pull(
          str(repo.path), f'https://{repo.server}/{repo.owner}/{repo.repo}.git', refspecs=repo.branch, force=True
        )
        dulwich.porcelain.clean(str(repo.path), str(repo.path))
        output('')
        output(f'Repo `{repo.name}` updated', style='green')
      except:
        shutil.rmtree(repo.path, ignore_errors=True)
        output(f'Failed to update repo {repo.name}', style='red')
  for c in REPO_DIR.glob('*/*/*'):
    repo_spec = tuple(c.parts[-3:])
    if repo_spec not in repos_in_use:
      shutil.rmtree(c, ignore_errors=True)
      output(f'Removed unused repo cache {c}')
  with open(REPO_DIR / 'last_update', 'w') as f:
    f.write(datetime.datetime.now().isoformat())
  for repo_name in config.repos:
    _complete_alias(repo_name)


def ensure_repo_updated():
  last_update_file = REPO_DIR / 'last_update'
  if not last_update_file.exists():
    if INTERACTIVE.get():
      choice = questionary.confirm(
        'The repo cache is never updated, do you want to update it to fetch the latest model list?'
      ).ask()
      if choice:
        update()
      return
    else:
      output(
        'The repo cache is never updated, please run `openllm repo update` to fetch the latest model list', style='red'
      )
      raise typer.Exit(1)
  last_update = datetime.datetime.fromisoformat(last_update_file.read_text().strip())
  if datetime.datetime.now() - last_update > UPDATE_INTERVAL:
    if INTERACTIVE.get():
      choice = questionary.confirm(
        'The repo cache is outdated, do you want to update it to fetch the latest model list?'
      ).ask()
      if choice:
        update()
    else:
      output(
        'The repo cache is outdated, please run `openllm repo update` to fetch the latest model list', style='yellow'
      )


GIT_REPO_RE = re.compile(r'git\+https://(?P<server>.+)/(?P<owner>.+)/(?P<repo>.+?)(@(?P<branch>.+))?$')


def parse_repo_url(repo_url, repo_name=None) -> RepoInfo:
  """
  parse the git repo url to server, owner, repo name, branch
  >>> parse_repo_url('git+https://github.com/bentoml/bentovllm@main')
  ('github.com', 'bentoml', 'bentovllm', 'main')

  >>> parse_repo_url('git+https://github.com/bentoml/bentovllm')
  ('github.com', 'bentoml', 'bentovllm', 'main')
  """
  match = GIT_REPO_RE.match(repo_url)
  if not match:
    raise ValueError(f'Invalid git repo url: {repo_url}')
  server = match.group('server')
  owner = match.group('owner')
  repo = match.group('repo')
  branch = match.group('branch') or 'main'
  path = REPO_DIR / server / owner / repo
  return RepoInfo(
    name=repo if repo_name is None else repo_name,
    url=repo_url,
    server=server,
    owner=owner,
    repo=repo,
    branch=branch,
    path=path,
  )


@app.command()
def add(name: str, repo: str):
  name = name.lower()
  if not name.isidentifier():
    output(f'Invalid repo name: {name}, should only contain letters, numbers and underscores', style='red')
    return

  config = load_config()
  if name in config.repos:
    override = questionary.confirm(f'Repo {name} already exists({config.repos[name]}), override?').ask()
    if not override:
      return

  config.repos[name] = repo
  save_config(config)
  output(f'Repo {name} added', style='green')


if __name__ == '__main__':
  app()
