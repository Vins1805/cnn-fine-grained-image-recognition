import time
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from datasets.dataset import CUBDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

torch.cuda.empty_cache()
print(torch.cuda.is_available())

vgg19 = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True).cuda()

trainset = CUBDataset()
trainloader = DataLoader(dataset=trainset, batch_size=10, shuffle=True)

for img, label in trainloader:
    pass

print(vgg19.features(img.cuda()))
    