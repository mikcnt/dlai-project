from typing import Any, Dict, Sequence, Tuple, Union
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
import torch.nn.functional as F

from src.common.utils import PROJECT_ROOT

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

from src.pl_modules.vae import VaeModel


def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    """Computes the gmm loss.
    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.
    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited
    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))
    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob


class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            in_features=hiddens, out_features=(2 * latents + 1) * gaussians + 2
        )

    def forward(self, *inputs):
        pass


class MDRNN(_MDRNNBase):
    """MDRNN model for multi steps forward"""

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents):  # pylint: disable=arguments-differ
        """MULTI STEPS forward.
        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)
        actions = actions.view(actions.shape[0], actions.shape[1], 1)
        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride : 2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride : 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds


class MDRNNCell(_MDRNNBase):
    """MDRNN model for one step forward"""

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden):  # pylint: disable=arguments-differ
        """ONE STEP forward.
        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor
        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride : 2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride : 2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden


class MDRNNModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!
        self.model = MDRNN(
            latents=self.hparams.LSIZE,
            actions=self.hparams.ASIZE,
            hiddens=self.hparams.RSIZE,
            gaussians=5,
        )

        self.vae = VaeModel.load_from_checkpoint(
            checkpoint_path=self.hparams.vae_checkpoint_path, map_location=self.device
        )
        self.vae.eval()

    def to_latent(self, obs, next_obs):
        """Transform observations to latent space.
        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        with torch.no_grad():
            obs, next_obs = [
                f.interpolate(
                    x.view(-1, 3, self.hparams.SIZE, self.hparams.SIZE),
                    size=self.hparams.RED_SIZE,
                    mode="bilinear",
                    align_corners=True,
                )
                for x in (obs, next_obs)
            ]
            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                self.vae(x)[1:] for x in (obs, next_obs)
            ]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(
                    self.hparams.BSIZE,
                    self.hparams.SEQ_LEN,
                    self.hparams.LSIZE,
                )
                for x_mu, x_logsigma in [
                    (obs_mu, obs_logsigma),
                    (next_obs_mu, next_obs_logsigma),
                ]
            ]
        return latent_obs, latent_next_obs

    def get_loss(
        self,
        latent_obs,
        action,
        reward,
        terminal,
        latent_next_obs,
        include_reward: bool,
    ):
        """Compute losses.
        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).
        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action, reward, terminal, latent_next_obs = [
            arr.transpose(1, 0)
            for arr in [latent_obs, action, reward, terminal, latent_next_obs]
        ]
        mus, sigmas, logpi, rs, ds = self.model(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = self.hparams.LSIZE + 2
        else:
            mse = 0
            scale = self.hparams.LSIZE + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

    def forward(self, batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        obs, action, reward, terminal, next_obs = batch.values()
        latent_obs, latent_next_obs = self.to_latent(obs, next_obs)
        losses = self.get_loss(
            latent_obs,
            action,
            reward,
            terminal,
            latent_next_obs,
            include_reward=True,
        )
        return losses

    def step(self, batch: Any, batch_idx: int):
        return self(batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)["loss"]

        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)["loss"]
        self.log_dict(
            {"val_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)["loss"]
        self.log_dict(
            {"test_loss": loss},
        )
        return loss

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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="mdrnn")
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
