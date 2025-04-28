from pathlib import Path

import jsonargparse
import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from safetensors.torch import save_file

from src.datamodule import DreamBoothDatamodule
from src.pl_module import LatentDiffusionModule


def main(parser: jsonargparse.ArgumentParser):
    cfg = parser.parse_args()
    L.seed_everything(cfg.seed)

    datamodule = DreamBoothDatamodule(**cfg.datamodule)

    pl_module = LatentDiffusionModule(**cfg.pl_module)

    # Modify checkpoint callback
    cfg_callbacks = cfg.trainer.pop("callbacks")
    cp_callback = ModelCheckpoint(
        save_weights_only=True,
        save_last=True,
    )

    trainer = Trainer(**cfg.trainer, callbacks=[cp_callback])

    print(f"Writing logs to {trainer.log_dir}")
    Path(trainer.log_dir).mkdir(parents=True)
    parser.save(cfg, Path(trainer.log_dir) / "config.yaml")

    trainer.fit(model=pl_module, datamodule=datamodule)

    # Convert last checkpoint to safetensors after training
    print("Best model:", cp_callback.best_model_path)
    ckpt = torch.load(cp_callback.best_model_path, weights_only=True)
    state_dict = ckpt["state_dict"]
    save_file(state_dict, Path(trainer.log_dir) / "lora_weights.safetensors")


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
            "instance_data_dir": "./data/instance_images",
            "instance_prompt": "a photo of sks dog",
            "with_prior_preservation": True,
            "class_data_dir": "./data/class_images",
            "class_prompt": "a photo of a dog",
        },
    )
    parser.add_class_arguments(LatentDiffusionModule, "pl_module")
    parser.link_arguments(
        "datamodule.pretrained_model_name_or_path", "pl_module.pretrained_model_name_or_path", apply_on="parse"
    )
    parser.link_arguments("datamodule.with_prior_preservation", "pl_module.with_prior_preservation", apply_on="parse")
    parser.add_class_arguments(
        Trainer,
        "trainer",
        default={
            "max_steps": 10,
            "log_every_n_steps": 1,
            "enable_checkpointing": True,
        },
    )

    main(parser)
