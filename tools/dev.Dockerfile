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
    -Isrc --grpc_python_out=${GENERATED_PB3_DIR} --python_out=${GENERATED_PB3_DIR} \
    --mypy_out=${GENERATED_PB3_DIR} --mypy_grpc_out=${GENERATED_PB3_DIR} \
    src/bentoml/grpc/${PROTOCOL_VERSION}/service.proto

mv ${GENERATED_PB3_DIR}/bentoml/grpc/${PROTOCOL_VERSION}/* /result/${GENERATED_PB3_DIR}
touch /result/${GENERATED_PB3_DIR}/__init__.py
rm -rf /result/${GENERATED_PB3_DIR}/${PROTOCOL_VERSION}

EOT

FROM scratch as protobuf-3-output

ARG GENERATED_PB3_DIR

COPY --from=run-grpcio-tools-3 /result/${GENERATED_PB3_DIR} /

############################################

# Triton stubs generation

FROM protobuf-3 as generate-triton-stubs

ARG GENERATED_DIR

RUN mkdir -p /result/${GENERATED_DIR}

RUN --mount=type=bind,target=.,rw <<EOT
set -ex

git clone --depth 1 --filter=blob:none --sparse https://github.com/triton-inference-server/common.git
cd common && git sparse-checkout set protobuf
cp protobuf/model_config.proto model_config.proto
cp protobuf/grpc_service.proto service.proto

mkdir -p ${GENERATED_DIR}

python -m grpc_tools.protoc -I. --mypy_out=${GENERATED_DIR} model_config.proto

python -m grpc_tools.protoc -I. --mypy_out=${GENERATED_DIR} --mypy_grpc_out=${GENERATED_DIR} service.proto

tree typings/tritonclient || exit 1

mv typings/tritonclient/grpc/* /result/${GENERATED_DIR}
EOT

FROM scratch as triton-protobuf-output

ARG GENERATED_DIR

COPY --from=generate-triton-stubs /result/${GENERATED_DIR} /

############################################

# Tests protobuf 3 generation

FROM protobuf-3 as generate-tests-proto-3

RUN mkdir -p /result/tests/proto/_generated_pb3

RUN --mount=type=bind,target=.,rw <<EOT
set -ex
mkdir -p tests/proto/_generated_pb3

python -m grpc_tools.protoc \
    -I. --grpc_python_out=tests/proto/_generated_pb3 --python_out=tests/proto/_generated_pb3 \
    --mypy_out=tests/proto/_generated_pb3 --mypy_grpc_out=tests/proto/_generated_pb3 \
    tests/proto/service_test.proto

mv tests/proto/_generated_pb3/tests/proto/* /result/tests/proto/_generated_pb3

touch /result/tests/proto/_generated_pb3/__init__.py

EOT

FROM scratch as generate-tests-proto-3-output

COPY --from=generate-tests-proto-3 /result/* /

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
    -Isrc --grpc_python_out=${GENERATED_PB4_DIR} --python_out=${GENERATED_PB4_DIR} \
    --mypy_out=${GENERATED_PB4_DIR} --mypy_grpc_out=${GENERATED_PB4_DIR} \
    src/bentoml/grpc/${PROTOCOL_VERSION}/service.proto

mv ${GENERATED_PB4_DIR}/bentoml/grpc/${PROTOCOL_VERSION}/* /result/${GENERATED_PB4_DIR}
touch /result/${GENERATED_PB4_DIR}/__init__.py
rm -rf /result/${GENERATED_PB4_DIR}/${PROTOCOL_VERSION}
EOT

FROM scratch as protobuf-4-output

ARG GENERATED_PB4_DIR

COPY --from=run-grpcio-tools-4 /result/${GENERATED_PB4_DIR} /

############################################

# Tests protobuf 4 generation

FROM protobuf-4 as generate-tests-proto-4

RUN mkdir -p /result/tests/proto/_generated_pb4

RUN --mount=type=bind,target=.,rw <<EOT
set -ex
mkdir -p tests/proto/_generated_pb4

python -m grpc_tools.protoc \
    -I. --grpc_python_out=tests/proto/_generated_pb4 --python_out=tests/proto/_generated_pb4 \
    --mypy_out=tests/proto/_generated_pb4 --mypy_grpc_out=tests/proto/_generated_pb4 \
    tests/proto/service_test.proto

mv tests/proto/_generated_pb4/tests/proto/* /result/tests/proto/_generated_pb4

touch /result/tests/proto/_generated_pb4/__init__.py

EOT

FROM scratch as generate-tests-proto-4-output

COPY --from=generate-tests-proto-4 /result/* /
