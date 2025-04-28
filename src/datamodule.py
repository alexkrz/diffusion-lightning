from pathlib import Path
from typing import Optional

import lightning as L
import torch
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DiffusionPipeline
from diffusers.training_utils import free_memory
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer


class DiffusionData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augment = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def prepare_data(self):
        load_dataset("cifar10")

    def train_dataloader(self):
        dataset = load_dataset("cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=4)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    """A simple dataset to prepare the prompts to generate class images on multiple GPUs."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class DreamBoothDatamodule(L.LightningDataModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        instance_data_dir: str,
        instance_prompt: str,
        resolution: int = 512,
        center_crop: bool = False,
        with_prior_preservation: bool = False,
        class_data_dir: str = "./data/class_images",
        num_class_images: int = 100,
        class_prompt: str = "a photo of a dog",
        sample_batch_size: int = 4,
        pre_compute_text_embeddings: bool = False,
        tokenizer_max_length: Optional[int] = None,
        train_batch_size: int = 2,
        dataloader_num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Generate class images if prior preservation is enabled.
        if self.hparams.with_prior_preservation:
            accelerator = Accelerator()
            class_images_dir = Path(self.hparams.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.hparams.num_class_images:
                print("Generating images on device: ", accelerator.device)
                torch_dtype = torch.float16 if accelerator.device.type in ("cuda", "xpu") else torch.float32
                pipeline = DiffusionPipeline.from_pretrained(
                    self.hparams.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.hparams.num_class_images - cur_class_images
                print(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.hparams.class_prompt, num_new_images)
                sample_dataloader = DataLoader(sample_dataset, batch_size=self.hparams.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)

                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                free_memory()

    def setup(self, stage):
        print(f"Preparing data for stage {stage}..")

        tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        if not self.hparams.pre_compute_text_embeddings:
            pre_computed_encoder_hidden_states = None
            validation_prompt_encoder_hidden_states = None
            validation_prompt_negative_prompt_embeds = None
            pre_computed_class_prompt_encoder_hidden_states = None
        else:
            raise NotImplementedError()

        self.train_dataset = DreamBoothDataset(
            instance_data_root=self.hparams.instance_data_dir,
            instance_prompt=self.hparams.instance_prompt,
            class_data_root=self.hparams.class_data_dir if self.hparams.with_prior_preservation else None,
            class_prompt=self.hparams.class_prompt,
            class_num=self.hparams.num_class_images,
            tokenizer=tokenizer,
            size=self.hparams.resolution,
            center_crop=self.hparams.center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
            tokenizer_max_length=self.hparams.tokenizer_max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.hparams.with_prior_preservation),
            num_workers=self.hparams.dataloader_num_workers,
        )
