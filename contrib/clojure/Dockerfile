# syntax=docker/dockerfile-upstream:master
FROM node:19-slim as builder

LABEL org.opencontainers.image.source="https://github.com/bentoml/OpenLLM"
LABEL org.opencontainers.image.authors="Leon Ikinger (GutZuFusss) <leon.ikinger@googlemail.com>"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
  apt-get update && apt-get install -y --no-install-recommends \
    build-essential openjdk-17-jdk curl git npm bash python3 python3-pip gcc python3-dev \
  && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://github.com/clojure/brew-install/releases/latest/download/posix-install.sh | bash -
RUN pip3 install openllm

WORKDIR /usr/src

RUN npm install -g pnpm

COPY ./package.json .

ENV PATH /usr/src/node_modules/.bin:$PATH

COPY . .

RUN pnpm install && pnpm run release

FROM nginx:alpine as base

COPY --from=builder /usr/src/public/ /usr/share/nginx/html
