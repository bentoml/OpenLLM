# syntax=docker/dockerfile-upstream:master
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM python:3.9-slim-bullseye as base-container

# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  ccache \
  curl \
  libssl-dev ca-certificates make \
  git && \
  rm -rf /var/lib/apt/lists/*


# Install required dependencies
COPY openllm-python/src src
COPY hatch.toml README.md CHANGELOG.md openllm-python/pyproject.toml ./

# Install all required dependencies
# We have to install autoawq first to avoid conflict with torch, then reinstall torch with vllm
# below
# pip install autoawq --no-cache-dir && \
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install --extra-index-url "https://huggingface.github.io/autogptq-index/whl/cu118/" \
  -v --no-cache-dir \
  "ray==2.6.0" "vllm==0.2.2" xformers && \
  pip install --no-cache-dir -e .

FROM base-container

ENTRYPOINT ["python3", "-m", "openllm"]
