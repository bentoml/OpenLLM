## OpenLLM CI/CD

> [!NOTE]
> All actions within this repository should always be locked to a specific version. We are using [ratchet](https://github.com/sethvargo/ratchet)
> for doing this via [this script](https://github.com/bentoml/OpenLLM/blob/main/tools/lock-actions.sh)

OpenLLM uses a GitHub Action to run all CI/CD workflows. It also use [pre-commit.ci](https://pre-commit.ci/) to run CI for all pre-commit hooks.

The folder structure of this are as follow:

```prolog
.
├── CODEOWNERS                # Code owners
├── CODE_OF_CONDUCT.md        # Code of conduct
├── ISSUE_TEMPLATE            # Contains issue templates
├── SECURITY.md               # Security policy
├── actions                   # Contains helpers script for all actions
├── assets                    # Contains static assets to be used throughout this repository
├── dependabot.yml            # Dependabot configuration
└── workflows
    ├── binary-releases.yml   # Build and publish binary releases
    ├── build.yml             # Self-hosted EC2 runners
    ├── ci.yml                # CI workflow
    ├── cleanup.yml           # Cache cleanup
    ├── build-pypi.yml        # Build PyPI packages
    ├── create-releases.yml   # Create GitHub releases
    ├── cron.yml              # Cron jobs
    └── release-notes.yml     # Generate release notes
```

> [!IMPORTANT]
> All of the following jobs will and should only be run within the BentoML organisation and this repository.

### Self-hosted EC2 runners

The workflow for self-hosted EC2 runners is located in [build.yml](/.github/workflows/build.yml).
This workflow is currently used for building OpenLLM base images that contains all compiled kernels
for serving. It will then be published to the following registry:

- GitHub Container Registry (`ghcr.io/bentoml/openllm`): This is where users can extend the base image
  with their own custom kernels or use as base for building Bentos

- AWS Elastic Container Registry (`public.ecr.aws/y5w8i4y6/bentoml/openllm`): This is where all Bento
  created with `openllm` will be using. This is purely for build optimisation on BentoCloud.

There are a few ways to trigger this workflow:

- Automatically triggered when a new commit is pushed to the `main` branch and tag release

- On pull request: This will be triggered manually when the label `00 - EC2 Build`

- On commit with the following `[ec2 build]`

### Wheel compilation

The workflow for wheel compilation is located in [build-pypi.yml](/.github/workflows/build-pypi.yml).

To speed up CI, opt in to the following label `02 - Wheel Build` on pull request or add `[wheel build]` to commit message.

### Binary releases

The workflow for binary releases is located in [binary-releases.yml](/.github/workflows/binary-releases.yml).

To speed up CI, opt in to the following label `03 - Standalone Build` on pull request or add `[binary build]` to commit message.
