from typing import Any, Dict, Sequence, Tuple, Union, List

import torchvision
import wandb
from torch import nn
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
import torch.nn.functional as F

from src.common.utils import PROJECT_ROOT


class Encoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(
            in_channels=img_channels, out_channels=32, kernel_size=(4, 4), stride=(2, 2)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2)
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2)
        )

        self.fc_mu = nn.Linear(in_features=2 * 2 * 256, out_features=latent_size)
        self.fc_logsigma = nn.Linear(in_features=2 * 2 * 256, out_features=latent_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class Decoder(nn.Module):
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=128, kernel_size=(5, 5), stride=(2, 2)
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2)
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(6, 6), stride=(2, 2)
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=32, out_channels=img_channels, kernel_size=(6, 6), stride=(2, 2)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, img_channels, latent_size):
        super().__init__()
        self.encoder = Encoder(img_channels=img_channels, latent_size=latent_size)
        self.decoder = Decoder(img_channels=img_channels, latent_size=latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


class VaeModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!
        self.model = VAE(self.hparams.img_channels, self.hparams.latent_size)
        self.recon_loss = nn.MSELoss()

    # def loss_function(self, recon_x, x, mu, logsigma):
    # """VAE loss function"""
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    ## 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    # return {"reconstruction_loss": BCE, "continuity_loss": KLD}

    def loss_function(self, recon_x, x, mu, logsigma):
        """VAE loss function"""
        BCE = self.recon_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = torch.mean(
            -0.5
            * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp(), dim=1),
            dim=0,
        )
        return {"reconstruction_loss": BCE, "continuity_loss": KLD}

    def get_image_examples(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> Sequence[wandb.Image]:
        """
        Given real and "fake" translated images, produce a nice coupled images to log

        :param real: the real images with shape [batch, channel, w, h]
        :param fake: the fake image with shape [batch, channel, w, h]

        :returns: a sequence of wandb.Image to log and visualize the performance
        """
        example_images = []
        for i in range(real.shape[0]):
            couple = torchvision.utils.make_grid(
                [real[i], fake[i]],
                nrow=2,
                normalize=True,
                scale_each=True,
                pad_value=1,
                padding=4,
            )
            example_images.append(
                wandb.Image(couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB")
            )
        return example_images

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(batch)

    def step(self, batch: Any, batch_idx: int):
        recon_batch, mu, logvar = self(batch)
        losses = self.loss_function(recon_batch, batch, mu, logvar)

        return {"recon_batch": recon_batch, "losses": losses}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        losses = self.step(batch, batch_idx)["losses"]
        recon_loss = losses["reconstruction_loss"]
        cont_loss = losses["continuity_loss"]
        train_loss = recon_loss + cont_loss
        self.log_dict(
            {
                "train_loss": train_loss,
                "train_recon_loss": recon_loss,
                "train_cont_loss": cont_loss,
            },
            on_step=True,
            prog_bar=False,
        )
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        output = self.step(batch, batch_idx)
        recon_batch = output["recon_batch"]
        images = self.get_image_examples(batch, recon_batch)
        losses = self.step(batch, batch_idx)["losses"]
        recon_loss = losses["reconstruction_loss"]
        cont_loss = losses["continuity_loss"]
        val_loss = recon_loss + cont_loss
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_recon_loss": recon_loss,
                "val_cont_loss": cont_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"val_loss": val_loss, "images": images}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        images = []

        for x in outputs:
            images.extend(x["images"])

        images = images[: self.hparams.logging.n_log_images]

        # ignore if it not a real validation epoch. The first one is not.
        print(f"Logged {len(images)} images for each category.")

        self.logger.experiment.log(
            {f"images_{self.current_epoch}": images},
        )

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        losses = self.step(batch, batch_idx)["losses"]
        test_loss = losses["reconstruction_loss"] + losses["continuity_loss"]
        self.log_dict(
            {"test_loss": test_loss},
        )
        return test_loss

    def configure_optimizers(
        self,
    ) -> Dict[str, Any]:
        # ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        # return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="vae")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
