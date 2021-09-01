from src.pl_data.dataset import GameEpisodeDataset
from src.pl_modules.controller_utils import rnn_adjust_parameters
from src.pl_modules.mdrnn import MDRNNCell
from src.plunder_standardize_colors import standardize_colors
import torch
import matplotlib.pyplot as plt
from src.pl_modules.vae import VaeModel
import numpy as np


def to_numpy(x: torch.Tensor):
    return x.cpu().detach().squeeze().permute(1, 2, 0).numpy()


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

    # load MDRNN
    rnn_state = torch.load(mdrnn_path, map_location=device)
    rnn_state = rnn_state["state_dict"]
    rnn_state = rnn_adjust_parameters(rnn_state)
    mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
    mdrnn.load_state_dict(rnn_state)

    batch = dataset[12]

    seq_obs = batch["obs"][0]  # (32, 3, 64, 64)
    obs = seq_obs[-1]  # (3, 64, 64)
    obs = obs.unsqueeze(0).to(device)

    recon_x, mu, logsigma = vae(obs)
    recon_recon_x, _, _ = vae(recon_x)

    obs_np = to_numpy(obs)
    recon_x_np = to_numpy(recon_x)

    # visualize reconstruction (test vae)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(obs_np)
    axarr[1].imshow(recon_x_np)
    plt.show()

    # test mdrnn
    action = batch["actions"][0][0].to(device).unsqueeze(0)
    next_obs = batch["next_obs"][0][0].to(device).unsqueeze(0)

    hidden = [torch.zeros(1, RSIZE, device=device) for _ in range(2)]

    mus, sigmas, logpi, r, d, next_hidden = mdrnn(action, mu, hidden)

    print("ao", mu.shape, logsigma.shape)

    print(mus.shape)
    print(sigmas.shape)
    print(logpi.shape)
    print(r.shape)
    print(d.shape)
    print(next_hidden[0].shape, next_hidden[1].shape)

    print(vae.model.decoder)