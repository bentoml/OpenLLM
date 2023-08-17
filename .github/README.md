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
    ├── clojure-frontend.yml  # Clojure frontend build
    ├── compile-pypi.yml      # Compile PyPI packages
    ├── create-releases.yml   # Create GitHub releases
    ├── cron.yml              # Cron jobs
    └── release-notes.yml     # Generate release notes
```

### Self-hosted EC2 runners

> [!IMPORTANT]
> This job will only be available for running within the BentoML repository.

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

### Clojure UI (Community-maintained)

> [!IMPORTANT]
> This job will only be available for running within the BentoML repository.

The workflow for Clojure UI is located in [clojure-frontend.yml](/.github/workflows/clojure-frontend.yml).
This workflow is currently used for building the Clojure UI and published to GitHub Container Registry (`ghcr.io/bentoml/openllm-ui-clojure`).

There are a few ways to trigger this workflow:

- This workflow will only trigger when there is a new `tag`

- On commit that contains `[clojure-ui build]` or Pull request with tag `01 - Clojure Build`
