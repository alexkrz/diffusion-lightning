from typing import Optional

from jsonargparse import CLI, Namespace
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.datamodule import DiffusionData, DreamBoothDataset, collate_fn
from src.pl_module import DiffusionModel


def parse_args(
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    instance_data_dir: str = "./data/dog",
    instance_promt: str = "a photo of sks dog",
    resolution: int = 512,
    revision: Optional[str] = None,
    pre_computed_text_embebeddings: Optional[str] = None,
    class_prompt: Optional[str] = None,
):
    """Script to train a diffusion model with Dreambooth and LoRA."""
    args = Namespace(locals())
    return args


def main(args):
    # Make some assertions
    assert args.pre_computed_text_embebeddings is None
    pre_computed_encoder_hidden_states = None
    assert args.class_prompt is None
    pre_computed_class_prompt_encoder_hidden_states = None

    # Dataset and DataLoaders creation:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        class_num=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=args.tokenizer_max_length,
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
    args = CLI(parse_args, as_positional=False)
    main(args)
