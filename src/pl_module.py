import diffusers
import lightning as L
import torch


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
