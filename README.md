> **Note:** This project requires **Python 3.11** or higher.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-0C3C4C.svg)](https://github.com/astral-sh/ruff)

# udl_project
This repository was created for the seminar "Understanding Deep Learning," which took place in the summer term of 2025 at the University of Osnabrück.

## What this repository shows

TODO: describe the novelty of the project

# project setup
.
├── Makefile (*project setup on linux based systems*)
├── README.md
├── artifacts (*place to store output from the project e.g. stored model*)
│   └── flower_classification_model.pth
├── infrastructure (*(pinned) dependencies*)
│   ├── requirements-dev.txt
│   ├── requirements.in
│   └── requirements.txt
├── pyproject.toml
└── udl_project (source code)


To create tree view run:
```bash
tree -I '.*|*data*'
```
Ignoring hidden and explicitly named files.

# Code Setup
We strongly recommend a Linux based system. If you are using Windows consider [installing WSL](https://learn.microsoft.com/de-de/windows/wsl/install) to use a Linux subsystem on Windows. We suggest using the latest Ubuntu.

✅ Make should come pre-installed.

🛠️ Install `uv` using [pipx as described in the official documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).

🚀 To set up the project, execute:
```bash
make setup
```

To set up the project on Windows, see the instructions in the [Code Setup](#code-setup) section above.

# Development Setup
We strongly recommend a Linux based system. If you are using Windows consider [installing WSL](https://learn.microsoft.com/de-de/windows/wsl/install) to use a Linux subsystem on Windows. We suggest using the latest Ubuntu.

✅ Make should come pre-installed.

🛠️ Install `uv` using [pipx as described in the official documentation](https://docs.astral.sh/uv/getting-started/installation/#pypi).

🚀 To set up the project, execute:
```bash
make setup-dev
```

This installs the additional development dependencies as listed in `infrastructure/requirements-dev.txt`.

To activate the pre-commit hooks, run:
```bash
pre-commit install
```

To run the pre-commit hooks manually, execute:
```bash
pre-commit run --all-files
```
or
```bash
make run-pc
```

To set up the project on Windows, see the instructions in the [Development Setup](#development-setup) section above.

# AI usage disclaimer

GitHub Copilot is being used in this project for the following purposes:

- Automatically reviewing Pull Requests, creating commit messages, descriptions.
- Suggestions for documentation
- Giving explanation on code snippets
- Support with arising bugs

The idea and the work on the code itself did not stem from GitHub copilot and is the work of the authors.
