# syntax=docker/dockerfile-upstream:master

FROM python:3.10-slim as base

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN --mount=type=cache,target=/var/lib/apt \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -q -y --no-install-recommends --allow-remove-essential \
        bash build-essential ca-certificates git tree

FROM base as protobuf-3

COPY <<-EOT requirements.txt
    protobuf>=3.5.0,<4.0dev
    grpcio-tools
    mypy-protobuf
EOT

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

FROM base as protobuf-4

COPY <<-EOT requirements.txt
    protobuf>=4.0,<5.0dev
    grpcio-tools
    mypy-protobuf
EOT

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

############################################

# BentoML gRPC protobuf 3 generation

FROM protobuf-3 as run-grpcio-tools-3

ARG PROTOCOL_VERSION
ARG GENERATED_PB3_DIR

RUN mkdir -p /result/${GENERATED_PB3_DIR}

RUN --mount=type=bind,target=.,rw <<EOT
set -ex

mkdir -p ${GENERATED_PB3_DIR}

python -m grpc_tools.protoc \
    -Iprotos  --grpc_python_out=${GENERATED_PB3_DIR} --python_out=${GENERATED_PB3_DIR} \
    --mypy_out=${GENERATED_PB3_DIR} --mypy_grpc_out=${GENERATED_PB3_DIR} \
    protos/service.proto

mv ${GENERATED_PB3_DIR}/* /result/${GENERATED_PB3_DIR}
touch /result/${GENERATED_PB3_DIR}/__init__.py
rm -rf /result/${GENERATED_PB3_DIR}/${PROTOCOL_VERSION}

EOT

FROM scratch as protobuf-3-output

ARG GENERATED_PB3_DIR

COPY --from=run-grpcio-tools-3 /result/${GENERATED_PB3_DIR} /

############################################

# BentoML gRPC protobuf 4 generation

FROM protobuf-4 as run-grpcio-tools-4

ARG PROTOCOL_VERSION
ARG GENERATED_PB4_DIR

RUN mkdir -p /result/${GENERATED_PB4_DIR}

RUN --mount=type=bind,target=.,rw <<EOT
set -ex

mkdir -p ${GENERATED_PB4_DIR}

python -m grpc_tools.protoc \
    -Iprotos --grpc_python_out=${GENERATED_PB4_DIR} --python_out=${GENERATED_PB4_DIR} \
    --mypy_out=${GENERATED_PB4_DIR} --mypy_grpc_out=${GENERATED_PB4_DIR} \
    protos/service.proto

mv ${GENERATED_PB4_DIR}/* /result/${GENERATED_PB4_DIR}
touch /result/${GENERATED_PB4_DIR}/__init__.py
rm -rf /result/${GENERATED_PB4_DIR}/${PROTOCOL_VERSION}
EOT

FROM scratch as protobuf-4-output

ARG GENERATED_PB4_DIR

COPY --from=run-grpcio-tools-4 /result/${GENERATED_PB4_DIR} /
