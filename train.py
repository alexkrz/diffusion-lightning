import logging
import os
from dataclasses import dataclass
from pathlib import Path

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from jsonargparse import CLI
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig

from src.datamodule import DreamBoothDataset, collate_fn

logger = get_logger(__name__)


@dataclass
class Args:
    """CLI for training a diffusion model using Dreambooth and LoRA."""

    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    instance_data_dir: str = "./data/dog"
    instance_prompt: str = "a photo of sks dog"
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"
    resolution: int = 512
    with_prior_preservation: bool = False
    train_batch_size: int = 4
    dataloader_num_workers: int = 0
    gradient_accumulation_steps: int = 1
    seed: int = 42


def main(args: Args):
    # Configure accelerator and logging
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Set up data
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

    # Set up model
    try:
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    except OSError:
        raise RuntimeError("Could not load text_encoder")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    try:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    except OSError:
        raise RuntimeError("Could not load vae")

    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)


if __name__ == "__main__":
    args = CLI(Args, as_positional=False)
    main(args)
