# syntax=docker/dockerfile-upstream:master
FROM node:19-alpine3.18 as builder

LABEL org.opencontainers.image.source="https://github.com/bentoml/OpenLLM"
LABEL org.opencontainers.image.authors="Leon Ikinger (GutZuFusss) <leon.ikinger@googlemail.com>"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN --mount=type=cache,target=/var/cache/apk \
    apk add --update openjdk17 curl git npm bash \
    && rm -rf /var/cache/apk/*
RUN curl -fsSL https://github.com/clojure/brew-install/releases/latest/download/posix-install.sh | bash -

WORKDIR /usr/src

RUN npm install -g pnpm

COPY ./package.json .

ENV PATH /usr/src/node_modules/.bin:$PATH

COPY . .

RUN pnpm install && pnpm run release

FROM nginx:alpine as base

COPY --from=builder /usr/src/public/ /usr/share/nginx/html
