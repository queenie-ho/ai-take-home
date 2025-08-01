import wandb
import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Union, Dict, Any, Tuple, Optional


class MNISTGANModel(LightningModule):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def adversial_loss(self, y_hat, y):
        return nn.MSELoss()(y_hat, y.float())

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        return [opt_g, opt_d], []

    def training_step(
        self, batch, batch_idx, optimizer_idx
    ) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        # TODO: if you have time, try implementing a test step
        raise NotImplementedError

    def step(self, batch, batch_idx, optimizer_idx=None):
        imgs, labels = batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        log_dict = {}
        loss = None

        # Train the generator 
        if optimizer_idx == 0 or not self.training:
            # Define sampling noise
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=device)
            y = torch.randint(0, 10, size=(imgs.shape[0],), device=device)

            # Generate images
            imgs_generated = self(z, y)

            # Classify generated image using the discriminator
            d_output = torch.squeeze(self.discriminator(imgs_generated, y))

            # Calculate adversarial loss
            valid = torch.ones(imgs.shape[0], device=device)
            g_loss = self.adversial_loss(d_output, valid)

            log_dict["g_loss"] = g_loss
            
            if optimizer_idx == 0:
                loss = g_loss

        
        # Train the discriminator
        if optimizer_idx == 1 or not self.training:
            d_output = torch.squeeze(self.discriminator(imgs, labels))

            # Calcualte loss for real img
            valid = torch.ones(imgs.shape[0], device=device)
            loss_real = self.adversial_loss(d_output, valid)

            # Create fake images
            z = torch.randn(imgs.shape[0], self.hparams.latent_dim, device=device)
            y = torch.randint(0, 10, size=(imgs.shape[0],), device=device)
            
            # Calcualte loss for fake img
            imgs_generated = self(z, y)
            d_output = torch.squeeze(self.discriminator(imgs_generated, y))
            loss_fake = self.adversial_loss(d_output, y)

            # Calculate average loss as discriminator loss
            d_loss = (loss_real + loss_fake) / 2
            
            log_dict["d_loss"] = d_loss
            log_dict["loss_real"] = loss_real
            log_dict["loss_fake"] = loss_fake
            
            if optimizer_idx == 1:
                loss = d_loss

        return log_dict, loss

    def on_epoch_end(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        z = torch.randn(8, self.hparams.latent_dim, device=device)
        y = torch.arange(0, 8, device=device)

        # Create image
        sample_imgs = self.generator(z, y)
        sample_imgs = sample_imgs.detach().cpu()
        sample_imgs = (sample_imgs + 1) / 2
        resize_img_size = int(np.sqrt(sample_imgs.shape[1]))
        sample_imgs = sample_imgs.reshape(8, resize_img_size, resize_img_size)

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                # Log fake images to wandb
                wandb_images = [
                    wandb.Image(img.unsqueeze(0), caption=f"Generated digit: {label}")
                    for img, label in zip(sample_imgs, y)
                ]
                logger.experiment.log({"gen_imgs": wandb_images})

