import diffusers
import lightning as L
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel


class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(sample_size=32)
        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


class LatentDiffusionModule(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        lora_rank: int = 4,
        learning_rate: float = 5e-4,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        lr_training_steps: int = 2,
        lr_num_cycles: int = 1,
        lr_power: float = 1.0,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
        self.unet.requires_grad_(False)
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        # Add LoRA adapter to UNet
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
        )
        self.unet.add_adapter(unet_lora_config)

        # Save hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        params_to_optimize = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            weight_decay=self.hparams.adam_weight_decay,
            eps=self.hparams.adam_epsilon,
        )

        lr_scheduler = diffusers.optimization.get_scheduler(
            name=self.hparams.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=self.hparams.lr_training_steps,
            num_cycles=self.hparams.lr_num_cycles,
            power=self.hparams.lr_power,
        )

        return [optimizer], [lr_scheduler]
