import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import Siamese_Net_dataset as SN_data
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
from PIL import Image
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = (256, 256)
transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
pic = cv2.imread('./BMP600/100_1.bmp')
pic = Image.fromarray(pic)
img = transform(pic)
img = img.view(1, 3, 256, 256)
img.to(DEVICE)

model = torchvision.models.resnet50(pretrained=False)
output = model(img)
print(output.size())
