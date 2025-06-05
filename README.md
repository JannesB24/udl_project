> **Note:** This project requires **Python 3.11** or higher.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-0C3C4C.svg)](https://github.com/astral-sh/ruff)

# udl_project
⚒️ *Work In Progress*

This repository was created for the seminar "[Understanding Deep Learning](https://udlbook.github.io/udlbook/)" which took place in the summer term of 2025 at the University of Osnabrück.

# What this repository shows
This project demonstrates a small residual network (ResNet) that initially struggles to accurately classify the dataset mentioned in the [prerequisites](#prerequisites). Various regularization techniques are applied and compared to improve its performance. The goal is to showcase a comprehensive comparison of different regularization techniques and their impact on the model's performance.

- ✅ **L2 Regularization** (Section 9.1.2 from [book](https://udlbook.github.io/udlbook/))
- 👷 **Early Stopping**
- ✅ **Ensembling** (Section 9.3.2 from [book](https://udlbook.github.io/udlbook/))
- 👷 **Dropout**
- 👷 **Applying noise**
- 👷 **...**

# Prerequisites
Download [this dataset from Kaggle](https://www.kaggle.com/datasets/lara311/flowers-five-classes).
Each user must download the dataset themselves, as it is not included in this repository for licensing reasons. The dataset must be present next to the project folder (`udl_project`) in the root directory of the project (`data/train/[label]`, e.g., daisy or rose).

## Project Setup
<pre>
.
├── Makefile (*project setup on Linux based systems*)
├── README.md
├── artifacts (*place to store output from the project e.g. stored model*)
│   └── flower_classification_model.pth
├── infrastructure (*(pinned) dependencies*)
│   ├── requirements-dev.txt
│   ├── requirements.in
│   └── requirements.txt
├── pyproject.toml
└── udl_project (source code)
</pre>

To create the tree view, run:
```bash
tree -I '.*|*data*'
```
Ignoring hidden and explicitly named files.

# Code Setup
We strongly recommend a Linux-based system. If you are using Windows, consider [installing WSL](https://learn.microsoft.com/de-de/windows/wsl/install) to use a Linux subsystem on Windows. We suggest using the latest Ubuntu.

✅ Make should come pre-installed.

🚀 To set up the project, execute:
```bash
make setup
```

This creates a local virtual environment (.venv) and installs the required dependencies as listed in `infrastructure/requirements.txt` into it.

To set up the project on Windows, see the instructions in the [Code Setup](#code-setup) section above.

## Running the Code
The project can be executed by running the `udl_project/master_comparison.py` script.

``` bash
python -m udl_project.master_comparison
```

## Troubleshooting (If Nothing Works)
If the convenience setup does not work, you can set up the project manually by following these instructions:
```bash
python -m venv .venv # assuming python is installed properly
source .venv/bin/activate
pip install --upgrade pip
pip install -r infrastructure/requirements.txt
```

## Development Setup
We strongly recommend a Linux based system. If you are using Windows consider [installing WSL](https://learn.microsoft.com/de-de/windows/wsl/install) to use a Linux subsystem on Windows. We suggest using the latest Ubuntu.

✅ Make should come pre-installed.

🛠️ Install `uv` using [pipx](https://docs.astral.sh/uv/getting-started/installation/#pypi) as described in the official documentation.

🚀 To set up the project, execute:
```bash
make setup-dev
```
This installs the additional development dependencies as listed in `infrastructure/requirements-dev.txt`.

To activate the pre-commit hooks, run:
```bash
pre-commit install
```
This will help you maintain clean code with each commit.

To run the pre-commit hooks manually, execute:
```bash
pre-commit run --all-files
```
or
```bash
make run-pc
```

To set up the project on Windows, see the instructions in the [Development Setup](#development-setup) section above.

## Help wanted for Windows setup
If you have gone through the setup and are still searching for a Windows solution, you have found the right place.

If you are truly interested in using Windows (my sincerest condolences, though AMD GPUs are also challenging in WSL) and manage to get the repository running, please help us by contributing to the project. If interested, look for the maintainer names on studip (no emails in repos 🤐) and contact them to get access.

We acknowledge the feedback that it probably will not work on Windows.

# AI usage disclaimer
GitHub Copilot is being used in this project for the following purposes:

- Automatically reviewing Pull Requests, creating commit messages, descriptions.
- Suggestions for documentation
- Giving explanation on code snippets
- Support with arising bugs

The idea and the work on the code itself did not stem from GitHub copilot and is the work of the authors.
