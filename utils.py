import torch
import torchvision
from matplotlib import pyplot as plt

def save_grid(images, path):
    grid = torchvision.utils.make_grid(images)
    torchvision.utils.save_image(grid, path)

def show_grid(images):
    grid = torchvision.utils.make_grid(images)
    plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.show()

def biggan_norm(images):
    images = (images + 1) / 2.0
    images = images.clip(0, 1)
    return images

def biggan_denorm(images):
    images = images*2 - 1
    return images


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

