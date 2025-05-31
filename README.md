> **Note:** This project requires **Python 3.11** or higher.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-0C3C4C.svg)](https://github.com/astral-sh/ruff)

# udl_project
⚒️ *Work In Progress*

This repository was created for the seminar "[Understanding Deep Learning](https://udlbook.github.io/udlbook/)" which took place in the summer term of 2025 at the University of Osnabrück.

# What this repository shows
This project demonstrates a small residual network (ResNet) that initially struggles to accurately classify the [dataset mentioned above](#prerequisites). Various regularization techniques are applied and compared to improve its performance. The goal is to showcase a comprehensive comparison of different regularization techniques and their impact on the model's performance.

👷 L2 Regularization
👷 Early Stopping
✅ Ensembling (Section 9.3.2 from [book](https://udlbook.github.io/udlbook/))
👷 Dropout
👷 Applying noise
👷 ...

# Prerequisites
Download [this dataset from Kaggle](https://www.kaggle.com/datasets/lara311/flowers-five-classes).
The dataset itself must be downloaded by each user themselves, since it is not included in this repository for licensing reasons.

## Project Setup
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


To create tree view run:
```bash
tree -I '.*|*data*'
```
Ignoring hidden and explicitly named files.

# Code Setup
We strongly recommend a Linux based system. If you are using Windows consider [installing WSL](https://learn.microsoft.com/de-de/windows/wsl/install) to use a Linux subsystem on Windows. We suggest using the latest Ubuntu.

✅ Make should come pre-installed.

🚀 To set up the project, execute:
```bash
make setup
```

This creates a local virtual environment (.venv) and installs the required dependencies as listed in `infrastructure/requirements.txt` into it.

To set up the project on Windows, see the instructions in the [Code Setup](#code-setup) section above.

## Running the Code
In the `udl_project` folder you find files starting with `train_`. These are the entry points and can be started with:
``` bash
python train_model.py
```

## AHHHHHH
To setup the project manually follow these instructions:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r infrastructure/requirements.txt
```

## Development Setup
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
If you looped through the setup still searching for a Windows setup you found the break and land here.

If you really are interested in using Windows (my sincerest condolences or well AMD GPU are no fun in WSL either) and use python on Windows and manage to get the repository running, please help us out by contributing to the project yourself.
In case you are interested, look for the maintainer names on studip (no e-mails in repos 🤐) and contact them ot get access.

We take the feedback, that it probably won't work on Windows.

# AI usage disclaimer
GitHub copilot is being used in this project for the following purposes:

- Automatically reviewing Pull Requests, creating commit messages, descriptions.
- Suggestions for documentation
- Giving explanation on code snippets
- Support with arising bugs

The idea and the work on the code itself did not stem from GitHub copilot and is the work of the authors.
