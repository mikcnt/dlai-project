from os.path import join

from torchvision.transforms import transforms
from tqdm import tqdm

from src.plunder_standardize_colors import standardize_colors
import numpy as np
import torch
from os import listdir, makedirs
import matplotlib.pyplot as plt
from src.pl_modules.vae import VaeModel
from PIL import Image


if __name__ == "__main__":

    images_path = "data/train/"
    reconstruction_path = "data/train_reconstruction/"
    makedirs(reconstruction_path, exist_ok=True)
    directories = listdir(images_path)
    directories = [join(images_path, x) for x in directories]

    files = []
    for i, directory in enumerate(directories):
        for file in listdir(directory):
            if file.startswith("img"):
                files.append(join(directory, file))
                break

    files = sorted(files)

    images = [np.load(file)["observations"] for file in files]

    for i, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(reconstruction_path + "/" + f"{i}".zfill(5) + ".png")

    # device = "cuda"
    # vae = VaeModel.load_from_checkpoint(
    #     "/home/michele/projects/dlai-project/checkpoints/vae/best.ckpt",
    #     map_location=device,
    # ).to(device)
    #
    # transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    #
    # for i, img in enumerate(tqdm(images)):
    #     obs = transform(img).cuda().unsqueeze(0)
    #     with torch.no_grad():
    #         reconstruction = vae(obs)[0]
    #     reconstruction = (
    #         reconstruction.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    #     )
    #     f, axarr = plt.subplots(2)
    #     axarr[0].imshow(img)
    #     axarr[1].imshow(reconstruction)
    #     f.savefig(reconstruction_path + "/" + f"{i}".zfill(5) + ".png")
    #     plt.close(f)
