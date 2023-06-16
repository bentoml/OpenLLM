# Security Policy

## Supported Versions

We are following [semantic versioning](https://semver.org/) with strict
backward-compatibility policy. We can ensure that all minor and major version
are backward compatible. We are more lenient with patch as the development can
move quickly.

If you are just using public API, then feel free to always upgrade. Whenever
there is a breaking policies, it will become a `DeprecationWarning` with a
period of 12 months before becoming broken.

> **Warning:** Everything package under `openllm` that has an underscore
> prefixes are exempt from this. They are considered private API and can change
> at any time. However, you can ensure that all public API, classes and
> functions will be backward-compatible.

## Reporting a Vulnerability

To report a security vulnerability, please send us an
[email](contact@bentoml.com).
