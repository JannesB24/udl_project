> **Note:** This project requires **Python 3.11** or higher.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-0C3C4C.svg)](https://github.com/astral-sh/ruff)

# udl_project
This repository was created for the seminar "Understanding Deep Learning," which took place in the summer term of 2025 at the University of Osnabrück.


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

This install the additional development dependencies as listed in `infrastructure/requirements.txt`.

To set up the project on Windows, see the instructions in the [Development Setup](#development-setup) section above.


