# Example from https://lightning.ai/lightning-ai/studios/train-a-diffusion-model-with-pytorch-lightning?section=featured
from jsonargparse import CLI
from lightning import Trainer

from src.datamodule import DiffusionData
from src.pl_module import DiffusionModel


def main():
    model = DiffusionModel()
    data = DiffusionData()
    trainer = Trainer(max_epochs=2, precision="bf16-mixed")
    trainer.fit(model, data)


if __name__ == "__main__":
    CLI(main, as_positional=False)
