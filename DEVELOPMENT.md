# Developer Guide

This Developer Guide is designed to help you contribute to the OpenLLM project.
Follow these steps to set up your development environment and learn the process
of contributing to our open-source project.

Join our [Discord Channel](https://l.bentoml.com/join-openllm-discord) and reach
out to us if you have any question!

## Table of Contents

- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
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
   hatch run dev:fmt
   ```
4. Write tests that verify your feature or fix (see
   [Writing Tests](#writing-tests) below).
5. Run all tests to ensure your changes haven't broken anything:

   ```bash
   hatch run test
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

## Writing Tests

Good tests are crucial for the stability of our codebase. Always write tests for
your features and fixes.

We use `pytest` for our tests. Make sure your tests are in the `tests/`
directory and their filenames start with `test_`.

Run all tests with:

```bash
hatch run test
```

## Releasing a New Version

To release a new version, use `gh workflow run`:

```bash
gh workflow run create-releases.yml
```

After the release CI finishes, then run the following:

```bash
gh workflow run release-notes.yml
```

> Note that currently this workflow can only be run by the BentoML team.
