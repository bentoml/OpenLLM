![Banner for OpenLLM](/.github/assets/main-banner.png)

<!-- hatch-fancy-pypi-readme intro start -->

<div align="center">
    <h1 align="center">👾 OpenLLM Client</h1>
    <a href="https://pypi.org/project/openllm-client">
        <img src="https://img.shields.io/pypi/v/openllm-client.svg?logo=pypi&label=PyPI&logoColor=gold" alt="pypi_status" />
    </a><a href="https://test.pypi.org/project/openllm-client/">
        <img src="https://img.shields.io/badge/Nightly-PyPI?logo=pypi&label=PyPI&color=gray&link=https%3A%2F%2Ftest.pypi.org%2Fproject%2Fopenllm%2F" alt="test_pypi_status" />
    </a><a href="https://twitter.com/bentomlai">
        <img src="https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow%20Us" alt="Twitter" />
    </a><a href="https://l.bentoml.com/join-openllm-discord">
        <img src="https://badgen.net/badge/icon/OpenLLM/7289da?icon=discord&label=Join%20Us" alt="Discord" />
    </a><a href="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml">
        <img src="https://github.com/bentoml/OpenLLM/actions/workflows/ci.yml/badge.svg?branch=main" alt="ci" />
    </a><a href="https://results.pre-commit.ci/latest/github/bentoml/OpenLLM/main">
        <img src="https://results.pre-commit.ci/badge/github/bentoml/OpenLLM/main.svg" alt="pre-commit.ci status" />
    </a><br>
    <a href="https://pypi.org/project/openllm-client">
        <img src="https://img.shields.io/pypi/pyversions/openllm-client.svg?logo=python&label=Python&logoColor=gold" alt="python_version" />
    </a><a href="htjtps://github.com/pypa/hatch">
        <img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" alt="Hatch" />
    </a><a href="https://github.com/bentoml/OpenLLM/blob/main/STYLE.md">
        <img src="https://img.shields.io/badge/code%20style-experimental-000000.svg" alt="code style" />
    </a><a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" alt="Ruff" />
    </a><a href="https://github.com/python/mypy">
        <img src="https://img.shields.io/badge/types-mypy-blue.svg" alt="types - mypy" />
    </a><a href="https://github.com/microsoft/pyright">
        <img src="https://img.shields.io/badge/types-pyright-yellow.svg" alt="types - pyright" />
    </a><br>
    <p>OpenLLM Client: Interacting with OpenLLM HTTP/gRPC server, or any BentoML server.<br/></p>
    <i></i>
</div>

## 📖 Introduction

With OpenLLM, you can run inference with any open-source large-language models,
deploy to the cloud or on-premises, and build powerful AI apps, and more.

To learn more about OpenLLM, please visit <a href="https://github.com/bentoml/OpenLLM">OpenLLM's README.md</a>

This package holds the underlying client implementation for OpenLLM. If you are
coming from OpenLLM, the client can be accessed via `openllm.client`.

```python
import openllm

client = openllm.client.HTTPClient()

client.query('Explain to me the difference between "further" and "farther"')
```

<!-- hatch-fancy-pypi-readme intro stop -->

![Gif showing OpenLLM Intro](/.github/assets/output.gif)

<br/>

<!-- hatch-fancy-pypi-readme interim start -->

## 📔 Citation

If you use OpenLLM in your research, we provide a [citation](../CITATION.cff) to use:

```bibtex
@software{Pham_OpenLLM_Operating_LLMs_2023,
author = {Pham, Aaron and Yang, Chaoyu and Sheng, Sean and  Zhao, Shenyang and Lee, Sauyon and Jiang, Bo and Dong, Fog and Guan, Xipeng and Ming, Frost},
license = {Apache-2.0},
month = jun,
title = {{OpenLLM: Operating LLMs in production}},
url = {https://github.com/bentoml/OpenLLM},
year = {2023}
}
```

<!-- hatch-fancy-pypi-readme interim stop -->
