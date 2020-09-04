# layer-to-layer-pytorch

<div align="center">

[![Build status](https://github.com/TezRomacH/layer-to-layer-pytorch/workflows/build/badge.svg?branch=master&event=push)](https://github.com/TezRomacH/layer-to-layer-pytorch/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/layer-to-layer-pytorch.svg)](https://pypi.org/project/layer-to-layer-pytorch/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/TezRomacH/layer-to-layer-pytorch/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/TezRomacH/layer-to-layer-pytorch/releases)
[![License](https://img.shields.io/github/license/TezRomacH/layer-to-layer-pytorch)](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/LICENSE)

PyTorch implementation of L2L execution algorithm
</div>

## 🚀 Features

For your development we've prepared:

- Supports for `Python 3.7` and higher.
- [`Poetry`](https://python-poetry.org/) as the dependencies manager. See configuration in [`pyproject.toml`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/pyproject.toml) and [`setup.cfg`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/setup.cfg).
- Power of [`black`](https://github.com/psf/black), [`isort`](https://github.com/timothycrosley/isort) and [`pyupgrade`](https://github.com/asottile/pyupgrade) formatters.
- Ready-to-use [`pre-commit`](https://pre-commit.com/) hooks with formatters above.
- Type checks with the configured [`mypy`](https://mypy.readthedocs.io).
- Testing with [`pytest`](https://docs.pytest.org/en/latest/).
- Docstring checks with [`darglint`](https://github.com/terrencepreilly/darglint).
- Security checks with [`safety`](https://github.com/pyupio/safety) and [`bandit`](https://github.com/PyCQA/bandit).
- Well-made [`.editorconfig`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.editorconfig), [`.dockerignore`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.dockerignore), and [`.gitignore`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.gitignore). You don't have to worry about those things.

For building and deployment:

- `GitHub` integration.
- [`Makefile`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/Makefile#L89) for building routines. Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc. More details at [Makefile summary](#makefile-usage)).
- [Dockerfile](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/docker/Dockerfile) for your package.
- `Github Actions` with predefined [build workflow](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.github/workflows/build.yml) as the default CI/CD.
- Always up-to-date dependencies with [`@dependabot`](https://dependabot.com/) (You will only [enable it](https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates)).
- Automatic drafts of new releases with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter). It creates a list of changes based on labels in merged `Pull Requests`. You can see labels (aka `categories`) in [`release-drafter.yml`](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.github/release-drafter.yml). Works perfectly with [Semantic Versions](https://semver.org/) specification.

For creating your open source community:

- Ready-to-use [Pull Requests templates](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.github/PULL_REQUEST_TEMPLATE.md) and several [Issue templates](https://github.com/TezRomacH/layer-to-layer-pytorch/tree/master/.github/ISSUE_TEMPLATE).
- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [`Stale bot`](https://github.com/apps/stale) that closes abandoned issues after a period of inactivity. (You will only [need to setup free plan](https://github.com/marketplace/stale)). Configuration is [here](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/.github/.stale.yml).
- [Semantic Versions](https://semver.org/) specification with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

## Installation [Not yet ready]

```bash
pip install layer-to-layer-pytorch
```

or install with `Poetry`

```bash
poetry add layer-to-layer-pytorch
```

## 📈 Releases

You can see the list of available releases on the [GitHub Releases](https://github.com/TezRomacH/layer-to-layer-pytorch/releases) page.

We follow [Semantic Versions](https://semver.org/) specification.

## 🛡 License

[![License](https://img.shields.io/github/license/TezRomacH/layer-to-layer-pytorch)](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/TezRomacH/layer-to-layer-pytorch/blob/master/LICENSE) for more details.

## 📃 Citation

```
@misc{layer-to-layer-pytorch,
  author = {Roman Tezikov},
  title = {PyTorch implementation of L2L execution algorithm},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TezRomacH/layer-to-layer-pytorch}}
}
```

## Credits

This project was generated with [`python-package-template`](https://github.com/TezRomacH/python-package-template).
