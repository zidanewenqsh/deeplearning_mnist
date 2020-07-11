import torch
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms

g_transform = transforms.Compose([
    transforms.Resize((170, 248)),
    transforms.ToTensor()
])


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def img2Tensor(imgpath)->torch.Tensor:
    with Image.open(imgpath) as img:
        return g_transform(img)