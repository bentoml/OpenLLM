load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//:__pkg__"])

[
    bzl_library(
        name = "{}.bzl".format(src),
        srcs = ["{}.bzl".format(src)],
    )
    for src in ["deps"]
]
