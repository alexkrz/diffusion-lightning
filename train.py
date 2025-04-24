from dataclasses import dataclass
from typing import Optional

from jsonargparse import CLI
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.datamodule import DiffusionData, DreamBoothDataset, collate_fn
from src.pl_module import DiffusionModel


@dataclass
class Args:
    """CLI for training a diffusion model using Dreambooth and LoRA."""

    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    instance_data_dir: str = "./data/dog"
    instance_prompt: str = "a photo of sks dog"
    resolution: int = 512
    with_prior_preservation: bool = False
    train_batch_size: int = 4
    dataloader_num_workers: int = 0


def main(args: Args):
    # Dataset and DataLoaders creation:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=args.resolution,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    model = DiffusionModel()
    data = DiffusionData()
    trainer = Trainer(max_epochs=2, precision="bf16-mixed")
    trainer.fit(model, data)


if __name__ == "__main__":
    args = CLI(Args, as_positional=False)
    main(args)
