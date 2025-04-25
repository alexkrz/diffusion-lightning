# Train DreamBooth with Lightning

Training a diffusion model with the [DreamBooth](https://dreambooth.github.io/) method using [Pytorch Lightning](https://lightning.ai/docs/overview/getting-started).

## Setup

We recommend [miniforge](https://conda-forge.org/download/) to set up your python environment.
In case VSCode does not detect your conda environments, install [nb_conda](https://github.com/conda-forge/nb_conda-feedstock) in the base environment.

```bash
conda env create -n $YOUR_ENV_NAME -f environment.yml
conda activate $YOUR_ENV_NAME
pip install -r requirements.txt
pre-commit install
```

## Props

- [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/)
- [diffusers](https://huggingface.co/docs/diffusers/index)
- [transformers](https://huggingface.co/docs/transformers/index)
- [lightning](https://lightning.ai/docs/overview/getting-started)

## Intention

This repository is meant to simplify the `train_dreambooth_lora.py` script found in the Huggingface Dreambooth instructions (<https://huggingface.co/docs/diffusers/training/dreambooth>).

The training can be executed by running

```bash
python train_lightning.py (--config configs/train_dreambooth_lightning.yaml)
```
