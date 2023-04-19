# TODO: Migrate to bzlmod once 6.0.0 is released.
workspace(name = "com_github_bentoml_bentoml")

load("//rules:deps.bzl", "bentoml_dependencies")

bentoml_dependencies()

load("@com_github_bentoml_plugins//rules:deps.bzl", "plugins_dependencies")

plugins_dependencies()

# NOTE: external users wish to use BentoML workspace setup
# should always be loaded in this order.
load("@com_github_bentoml_plugins//rules:workspace0.bzl", "workspace0")

workspace0()

load("@com_github_bentoml_plugins//rules:workspace1.bzl", "workspace1")

workspace1()

load("@com_github_bentoml_plugins//rules:workspace2.bzl", "workspace2")

workspace2()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    requirements = "//requirements:bazel-requirements.lock.txt",
)

pip_parse(
    name = "tensorflow",
    requirements = "//requirements:bazel-tensorflow-requirements.lock.txt",
)

pip_parse(
    name = "tests",
    requirements = "//requirements:bazel-tests-requirements.lock.txt",
)

load("//rules/py/vendorred:pypi.bzl", pypi_deps = "install_deps")

pypi_deps()

load("//rules/py/vendorred:tests.bzl", tests_deps = "install_deps")

tests_deps()

load("//rules/py/vendorred:tensorflow.bzl", tensorflow_deps = "install_deps")

tensorflow_deps()
