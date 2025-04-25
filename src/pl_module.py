import diffusers
import lightning as L
import torch
import torch.nn.functional as F
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


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


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
        with_prior_preservation: bool = False,
        prior_loss_weight: float = 1.0,
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

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]

        # Convert images to latent space
        model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = encode_prompt(
            self.text_encoder,
            batch["input_ids"],
            batch["attention_mask"],
            text_encoder_use_attention_mask=None,
        )

        if self.unet.config.in_channels == channels * 2:
            noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

        # Predict the noise residual
        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            encoder_hidden_states,
            class_labels=None,
            return_dict=False,
        )[0]

        # if model predicts variance, throw away the prediction. we will only train on the
        # simplified training objective. This means that all schedulers using the fine tuned
        # model must be configured to use one of the fixed variance variance types.
        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # TODO: Check what else to adjust when training with prior_preservation
        if self.hparams.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = loss + self.hparams.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        self.log("train_loss", loss, prog_bar=True)

        return loss

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
