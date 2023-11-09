# syntax=docker/dockerfile-upstream:master
FROM node:19-slim as builder

LABEL org.opencontainers.image.source="https://github.com/bentoml/OpenLLM"
LABEL org.opencontainers.image.authors="Leon Ikinger (GutZuFusss) <leon.ikinger@googlemail.com>"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

# vendorred from https://github.com/rust-lang/docker-rust/blob/d2002ab9f7a29ce1dcbbac7d77fbd598fd971d1a/1.72.0/buster/slim/Dockerfile
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.72.0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
        libc6-dev \
        wget \
        ; \
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
        amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='0b2f6c8f85a3d02fde2efc0ced4657869d73fccfce59defb4e8d29233116e6db' ;; \
        armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='f21c44b01678c645d8fbba1e55e4180a01ac5af2d38bcbd14aa665e0d96ed69a' ;; \
        arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='673e336c81c65e6b16dcdede33f4cc9ed0f08bde1dbe7a935f113605292dc800' ;; \
        i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='e7b0f47557c1afcd86939b118cbcf7fb95a5d1d917bdd355157b63ca00fc4333' ;; \
        *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
    url="https://static.rust-lang.org/rustup/archive/1.26.0/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version; \
    apt-get remove -y --auto-remove \
        wget \
        ; \
    rm -rf /var/lib/apt/lists/*;

RUN rustup target add aarch64-apple-darwin

RUN apt-get update && apt-get install -y --no-install-recommends \
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
