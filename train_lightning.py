import jsonargparse
import lightning as L
from lightning import Trainer

from src.datamodule import DreamBoothDatamodule
from src.pl_module import LatentDiffusionModule


def main(cfg: jsonargparse.Namespace):
    L.seed_everything(cfg.seed)

    datamodule = DreamBoothDatamodule(**cfg.datamodule)

    pl_module = LatentDiffusionModule(**cfg.pl_module)

    trainer = Trainer(**cfg.trainer)

    trainer.fit(model=pl_module, datamodule=datamodule)

    # TODO: Save LoRA weights after training


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_class_arguments(
        DreamBoothDatamodule,
        "datamodule",
        default={
            "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "instance_data_root": "./data/dog",
            "instance_prompt": "a photo of sks dog",
        },
    )
    parser.add_class_arguments(LatentDiffusionModule, "pl_module")
    parser.link_arguments(
        "datamodule.pretrained_model_name_or_path", "pl_module.pretrained_model_name_or_path", apply_on="parse"
    )
    parser.add_class_arguments(
        Trainer,
        "trainer",
        default={
            "max_epochs": 5,
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
        },
    )

    cfg = parser.parse_args()
    main(cfg)
