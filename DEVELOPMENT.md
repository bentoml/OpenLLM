# Developer Guide

This Developer Guide is designed to help you contribute to the OpenLLM project.
Follow these steps to set up your development environment and learn the process
of contributing to our open-source project.

Join our [Discord Channel](https://l.bentoml.com/join-openllm-discord) and reach
out to us if you have any question!

## Table of Contents

- [Developer Guide](#developer-guide)
  - [Table of Contents](#table-of-contents)
  - [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Project Structure](#project-structure)
  - [Development Workflow](#development-workflow)
  - [Using a custom fork](#using-a-custom-fork)
  - [Writing Tests](#writing-tests)
  - [Releasing a New Version](#releasing-a-new-version)

## Setting Up Your Development Environment

Before you can start developing, you'll need to set up your environment:

1. Ensure you have [Git](https://git-scm.com/), and
   [Python3.8+](https://www.python.org/downloads/) installed.
2. Fork the OpenLLM repository from GitHub.
3. Clone the forked repository from GitHub:

   ```bash
   git clone git@github.com:username/OpenLLM.git && cd openllm
   ```

4. Add the OpenLLM upstream remote to your local OpenLLM clone:

   ```bash
   git remote add upstream git@github.com:bentoml/OpenLLM.git
   ```

5. Configure git to pull from the upstream remote:

   ```bash
   git switch main # ensure you're on the main branch
   git fetch upstream --tags
   git branch --set-upstream-to=upstream/main
   ```

6. Install [hatch](https://github.com/pypa/hatch):

   ```bash
   pip install hatch pre-commit
   ```

7. Run the following to setup all pre-commit hooks:

   ```bash
   hatch run setup
   ```

8. Enter a project's environment with.
   ```bash
   hatch shell
   ```

   This will automatically enter a virtual environment and update the relevant
   dependencies.

## Project Structure

Here's a high-level overview of our project structure:

```
openllm/
├── examples                 # Usage demonstration scripts
├── src
│   ├── openllm              # Core OpenLLM library
│   ├── openllm_client       # OpenLLM Python Client code
│   └── openllm_js           # OpenLLM JavaScript Client code
├── tests                    # Automated Tests
├── tools                    # Utilities Script
├── typings                  # Typing Checking Utilities Module and Classes
├── DEVELOPMENT.md           # The project's Developer Guide
├── LICENSE                  # Use terms and conditions
├── package.json             # Node.js or JavaScript dependencies
├── pyproject.toml           # Python Project Specification File (PEP 518)
└── README.md                # The project's README file
```

## Development Workflow

After setting up your environment, here's how you can start contributing:

1. Create a new branch for your feature or fix:

   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes to the codebase.
3. Run all formatter and linter with `hatch`:

   ```bash
   hatch run quality
   ```
4. Write tests that verify your feature or fix (see
   [Writing Tests](#writing-tests) below).
5. Run all tests to ensure your changes haven't broken anything:

   ```bash
   hatch run tests:python
   ```
6. Commit your changes:

   ```bash
   git commit -m "Add my feature"
   ```
7. Push your changes to your fork:

   ```bash
   git push origin feature/my-feature
   ```
8. Submit a Pull Request on GitHub.

## Using a custom fork

If you wish to use a modified version of OpenLLM, install your fork from source
with `pip install -e` and set `OPENLLM_DEV_BUILD=True`, so that Bentos built
will include the generated wheels for OpenLLM in the bundle.

## Writing Tests

Good tests are crucial for the stability of our codebase. Always write tests for
your features and fixes.

We use `pytest` for our tests. Make sure your tests are in the `tests/`
directory and their filenames start with `test_`.

Run all tests with:

```bash
hatch run tests:python
```

Run snapshot testing for model outputs:

```bash
hatch run tests:models
```

To update the snapshot, do the following:

```bash
hatch run tests:snapshot-models
```

## Releasing a New Version

To release a new version, use `./tools/run-release-action`. It requires `gh`,
`jq` and `hatch`:

```bash
./tools/run-release-action --release <major|minor|patch>
```

> Note that currently this workflow can only be run by the BentoML team.

## Changelog

_modeled after the [attrs](https://github.com/python-attrs/attrs) workflow_

If the change is noteworthy, there needs to be a changelog entry so users can
learn about it!

To avoid merge conflicts, we use the
[_Towncrier_](https://pypi.org/project/towncrier) package to manage our
changelog. _towncrier_ uses independent _Markdown_ files for each pull request –
so called _news fragments_ – instead of one monolithic changelog file. On
release, those news fragments are compiled into
[`CHANGELOG.md`](https://github.com/bentoml/openllm/blob/main/CHANGELOG.md).

You don't need to install _Towncrier_ yourself, you just have to abide by a few
simple rules:

- For each pull request, add a new file into `changelog.d` with a filename
  adhering to the `<pr#>.(change|deprecation|breaking|feature).md` schema: For
  example, `changelog.d/42.change.md` for a non-breaking change that is proposed
  in pull request #42.
- As with other docs, please use [semantic newlines] within news fragments.
- Wrap symbols like modules, functions, or classes into backticks so they are
  rendered in a `monospace font`.
- Wrap arguments into asterisks like in docstrings:
  `Added new argument *an_argument*.`
- If you mention functions or other callables, add parentheses at the end of
  their names: `openllm.func()` or `openllm.LLMClass.method()`. This makes the
  changelog a lot more readable.
- Prefer simple past tense or constructions with "now". For example:

  - Added `LLM.func()`.
  - `LLM.func()` now doesn't do X.Y.Z anymore when passed the _foobar_ argument.
- If you want to reference multiple issues, copy the news fragment to another
  filename. _Towncrier_ will merge all news fragments with identical contents
  into one entry with multiple links to the respective pull requests.

Example entries:

```md
Added `LLM.func()`. The feature really _is_ awesome.
```

or:

```md
`openllm.utils.func()` now doesn't X.Y.Z anymore when passed the _foobar_
argument. The bug really _was_ nasty.
```

---

`hatch run changelog` will render the current changelog to the terminal if you
have any doubts.

[semantic newlines]: https://rhodesmill.org/brandon/2012/one-sentence-per-line/
