import numpy as np
import os
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import mnist1d
import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn.functional as F
import torch.optim as optim



#shamelessly copied from here:https://www.kaggle.com/code/kimkijun7/flower-classification-pytorch-with-python
train_dir = 'Flower Classification Dataset/train'
test_dir = 'Flower Classification Dataset/test'
val_dir = 'Flower Classification Dataset/valid'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 128

train_ds = torchvision.datasets.ImageFolder(
    train_dir, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True
)


val_ds = torchvision.datasets.ImageFolder(
    val_dir, transform=transform 
)

val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=batch_size, shuffle=True
)


test_ds = torchvision.datasets.ImageFolder(
    train_dir, transform=transform 
)

test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=True
)

