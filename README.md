# vscode-python-template

A Template for Python Programming in VSCode

## Setup

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment.
In case VSCode does not detect your conda environments, install [nb_conda](https://github.com/conda-forge/nb_conda-feedstock) in the base environment.

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```
