# syntax=docker/dockerfile-upstream:master
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as base-container

# Automatically set by buildx
ARG TARGETPLATFORM

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  ccache \
  curl \
  libssl-dev ca-certificates make \
  git python3-pip && \
  rm -rf /var/lib/apt/lists/*


# Install required dependencies
COPY openllm-python/src src
COPY hatch.toml README.md CHANGELOG.md openllm-python/pyproject.toml ./

# Install all required dependencies
# We have to install autoawq first to avoid conflict with torch, then reinstall torch with vllm
# below
RUN --mount=type=cache,target=/root/.cache/pip \
  pip3 install -v --no-cache-dir \
  "ray==2.6.0" "vllm==0.2.2" xformers && \
  pip3 install --no-cache-dir -e .

FROM base-container

ENTRYPOINT ["python3", "-m", "openllm"]
