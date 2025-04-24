import jsonargparse
from lightning import LightningDataModule, LightningModule, Trainer

from src.datamodule import DiffusionData, DreamBoothDatamodule
from src.pl_module import DiffusionModel


def process_parser_args(parser: jsonargparse.ArgumentParser):
    cfg = parser.parse_args()
    return cfg


def main(cfg):
    datamodule = DreamBoothDatamodule(**cfg.datamodule)

    model = DiffusionModel()
    data = DiffusionData()
    trainer = Trainer(max_epochs=2, precision="bf16-mixed")
    trainer.fit(model, data)


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--datamodule", type=DreamBoothDatamodule)
    # parser.add_argument("--pl_module", type=LightningModule)
    # parser.add_argument("--trainer", type=Trainer)
    parser.add_class_arguments(
        DreamBoothDatamodule,
        "datamodule",
        default={
            "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "instance_data_root": "./data/dog",
            "instance_prompt": "a photo of sks dog",
        },
    )

    cfg = process_parser_args(parser)

    # datamodule = parser.instantiate_classes({"datamodule": cfg.datamodule}).datamodule
    # pl_module = parser.instantiate_classes({"pl_module": cfg.pl_module}).pl_module

    main(cfg)
