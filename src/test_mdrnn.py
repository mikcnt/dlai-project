from src.pl_data.dataset import GameEpisodeDataset
from src.pl_modules.controller_utils import rnn_adjust_parameters
from src.pl_modules.mdrnn import MDRNNCell, MDRNN, MDRNNModel
from src.plunder_standardize_colors import standardize_colors
import torch
import matplotlib.pyplot as plt
from src.pl_modules.vae import VaeModel
import numpy as np
import torch.nn.functional as f


def to_numpy(x: torch.Tensor):
    return x.cpu().detach().squeeze().permute(1, 2, 0).numpy()


def to_latent(obs, next_obs, vae):
    bs, seq_len, channels, size, size = obs.shape

    with torch.no_grad():
        obs, next_obs = [
            f.interpolate(
                x.view(-1, 3, size, size),
                size=size,
                mode="bilinear",
                align_corners=True,
            )
            for x in (obs, next_obs)
        ]
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)
        ]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(bs, seq_len, 32,)
            for x_mu, x_logsigma in [
                (obs_mu, obs_logsigma),
                (next_obs_mu, next_obs_logsigma),
            ]
        ]
    return latent_obs, latent_next_obs


if __name__ == "__main__":
    ASIZE = 1  # Action size: 1 is a scalar, n is an n-tuple.
    LSIZE = 32  # Latent Size
    RSIZE = 256  # Number of hidden units
    RED_SIZE = 64  #
    SIZE = 64  # Image shape is SIZE x SIZE
    SEQ_LEN = 32  # Sequence length
    BSIZE = 32  # Batch size

    obs_path = "data/train/"
    dataset = GameEpisodeDataset(name=".", path=obs_path)

    vae_path = "checkpoints/vae/best.ckpt"
    mdrnn_path = "checkpoints/mdrnn/best.ckpt"

    device = "cuda"

    # load vae
    vae = VaeModel.load_from_checkpoint(vae_path, map_location=device).to(device)
    decoder = vae.model.decoder

    # load MDRNN
    mdrnn_model = MDRNNModel.load_from_checkpoint(mdrnn_path, map_location=device).to(
        device
    )
    mdrnn = mdrnn_model.model
    # rnn_state = torch.load(mdrnn_path, map_location=device)
    # rnn_state = rnn_state["state_dict"]
    # rnn_state = rnn_adjust_parameters(rnn_state)
    # mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5).to(device)
    # mdrnn.load_state_dict(rnn_state)

    batch = dataset[12]

    seq_obs = batch["obs"][0]  # (32, 3, 64, 64)
    obs = seq_obs[-1]  # (3, 64, 64)
    obs = obs.unsqueeze(0).to(device)

    recon_x, mu, logsigma = vae(obs)
    recon_recon_x, _, _ = vae(recon_x)

    obs_np = to_numpy(obs)
    recon_x_np = to_numpy(recon_x)

    # visualize reconstruction (test vae)
    # f, axarr = plt.subplots(2)
    # axarr[0].imshow(obs_np)
    # axarr[1].imshow(recon_x_np)
    # plt.show()

    # test mdrnn
    action = batch["actions"][0][0].to(device).unsqueeze(0)
    next_obs = batch["next_obs"][0][0].to(device).unsqueeze(0).unsqueeze(0)
    obs = obs.unsqueeze(0)


    latent_obs, latent_next_obs = to_latent(obs, next_obs, vae)

    # hidden = [torch.zeros(1, RSIZE, device=device) for _ in range(2)]

    # mus, sigmas, logpi, r, d, next_hidden = mdrnn(action, mu, hidden)
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    print("mu.shape =", mu.shape)
    print("logsigma.shape =", logsigma.shape)

    print("mus.shape =", mus.shape)
    print("sigmas.shape =", sigmas.shape)
    print("logpi.shape =", logpi.shape)
    # print("r.shape =", r.shape)
    # print("d.shape =", d.shape)

    print((logpi * torch.normal(mus, sigmas)).shape)
    sampled = torch.sum(logpi * torch.normal(mus, sigmas), dim=2)
