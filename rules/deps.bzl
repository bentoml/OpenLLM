load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

# NOTE: sync with pyproject.toml
GRPC_VERSION = "1.51.1"
GRPC_SHA256 = "b55696fb249669744de3e71acc54a9382bea0dce7cd5ba379b356b12b82d4229"
PROTOBUF_VERSION = "21.11"
PROTOBUF_SHA256 = "b1d6dd2cbb5d87e17af41cadb720322ce7e13af826268707bd8db47e5654770b"

def bentoml_dependencies():
    # bentoml/plugins
    maybe(
        git_repository,
        name = "com_github_bentoml_plugins",
        remote = "https://github.com/bentoml/plugins.git",
        branch = "main",
    )

    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "io_bazel_rules_go",
        sha256 = "d6b2513456fe2229811da7eb67a444be7785f5323c6708b38d851d2b51e54d83",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.30.0/rules_go-v0.30.0.zip",
            "https://github.com/bazelbuild/rules_go/releases/download/v0.30.0/rules_go-v0.30.0.zip",
        ],
    )

    maybe(
        http_archive,
        name = "io_bazel_rules_docker",
        sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
        urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
    )

    maybe(
        http_archive,
        name = "bazel_gazelle",
        sha256 = "de69a09dc70417580aabf20a28619bb3ef60d038470c7cf8442fafcf627c21cb",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_proto",
        sha256 = "80d3a4ec17354cccc898bfe32118edd934f851b03029d63ef3fc7c8663a7415c",
        strip_prefix = "rules_proto-5.3.0-21.5",
        urls = [
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.5.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_proto_grpc",
        strip_prefix = "rules_proto_grpc-4.2.0",
        sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
        urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
    )

    maybe(
        http_archive,
        name = "com_google_protobuf",
        strip_prefix = "protobuf-{}".format(PROTOBUF_VERSION),
        sha256 = PROTOBUF_SHA256,
        urls = [
            "https://github.com/protocolbuffers/protobuf/archive/v{}.tar.gz".format(PROTOBUF_VERSION),
        ],
    )

    maybe(
        http_archive,
        name = "com_github_grpc_grpc",
        strip_prefix = "grpc-{}".format(GRPC_VERSION),
        sha256 = GRPC_SHA256,
        urls = [
            "https://github.com/grpc/grpc/archive/v{}.tar.gz".format(GRPC_VERSION),
        ],
    )

    maybe(
        http_archive,
        name = "rules_foreign_cc",
        sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
        strip_prefix = "rules_foreign_cc-0.9.0",
        url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.9.0.tar.gz",
    )

    # buildifier
    maybe(
        http_archive,
        name = "com_github_bazelbuild_buildtools",
        sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
        strip_prefix = "buildtools-4.2.2",
        urls = [
            "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
        ],
    )

    # buf rules
    maybe(
        http_archive,
        name = "rules_buf",
        sha256 = "523a4e06f0746661e092d083757263a249fedca535bd6dd819a8c50de074731a",
        strip_prefix = "rules_buf-0.1.1",
        urls = [
            "https://github.com/bufbuild/rules_buf/archive/refs/tags/v0.1.1.zip",
        ],
    )

    # python rules
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "8c15896f6686beb5c631a4459a3aa8392daccaab805ea899c9d14215074b60ef",
        strip_prefix = "rules_python-0.17.3",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.17.3.tar.gz",
    )

    # The following library will need to be built from source.
    maybe(
        new_git_repository,
        name = "com_github_microsoft_lightgbm",
        init_submodules = True,
        recursive_init_submodules = True,
        commit = "f1d3181ced9fd01f4b2899054abd99be6773e939",
        build_file = Label("//third_party:BUILD.lightgbm"),
        remote = "https://github.com/microsoft/LightGBM.git",
        shallow_since = "1667710116 -0500",
    )

    # io_grpc_grpc_java is for java_grpc_library and related dependencies.
    # Using commit 0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8
    maybe(
        http_archive,
        name = "rules_jvm_external",
        sha256 = "c21ce8b8c4ccac87c809c317def87644cdc3a9dd650c74f41698d761c95175f3",
        strip_prefix = "rules_jvm_external-1498ac6ccd3ea9cdb84afed65aa257c57abf3e0a",
        url = "https://github.com/bazelbuild/rules_jvm_external/archive/1498ac6ccd3ea9cdb84afed65aa257c57abf3e0a.zip",
    )
    maybe(
        http_archive,
        name = "io_grpc_grpc_java",
        sha256 = "35189faf484096c9eb2928c43b39f2457d1ca39046704ba8c65a69482f8ceed5",
        strip_prefix = "grpc-java-0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8",
        urls = ["https://github.com/grpc/grpc-java/archive/0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8.tar.gz"],
    )

    # rules_kotlin
    maybe(
        http_archive,
        name = "io_bazel_rules_kotlin",
        sha256 = "a57591404423a52bd6b18ebba7979e8cd2243534736c5c94d35c89718ea38f94",
        urls = ["https://github.com/bazelbuild/rules_kotlin/releases/download/v1.6.0/rules_kotlin_release.tgz"],
    )
    maybe(
        http_archive,
        name = "com_github_grpc_grpc_kotlin",
        sha256 = "b1ec1caa5d81f4fa4dca0662f8112711c82d7db6ba89c928ca7baa4de50afbb2",
        strip_prefix = "grpc-kotlin-a1659c1b3fb665e01a6854224c7fdcafc8e54d56",
        urls = ["https://github.com/grpc/grpc-kotlin/archive/a1659c1b3fb665e01a6854224c7fdcafc8e54d56.tar.gz"],
    )

    # rules_swift and rules_apple
    maybe(
        http_archive,
        name = "build_bazel_rules_swift",
        sha256 = "51efdaf85e04e51174de76ef563f255451d5a5cd24c61ad902feeadafc7046d9",
        url = "https://github.com/bazelbuild/rules_swift/releases/download/1.2.0/rules_swift.1.2.0.tar.gz",
    )
    maybe(
        http_archive,
        name = "build_bazel_apple_support",
        sha256 = "2e3dc4d0000e8c2f5782ea7bb53162f37c485b5d8dc62bb3d7d7fc7c276f0d00",
        url = "https://github.com/bazelbuild/apple_support/releases/download/1.3.2/apple_support.1.3.2.tar.gz",
    )
