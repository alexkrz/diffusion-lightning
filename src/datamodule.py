import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
