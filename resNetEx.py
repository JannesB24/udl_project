import numpy as np
import os
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import mnist1d
import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn.functional as F
import torch.optim as optim
import DataloaderFlowerSet
import Models
import ResnetThief
#fck cuda
device=  torch.device('cpu')

image_batch =DataloaderFlowerSet.train_loader
def weights_init(layer_in):
  if isinstance(layer_in, nn.Linear):
    nn.init.kaiming_uniform_(layer_in.weight)
    layer_in.bias.data.fill_(0.0)

def run_one_step_of_model(model):
  # choose cross entropy loss function (equation 5.24 in the loss notes)
  loss_function = nn.CrossEntropyLoss()
  # construct SGD optimizer and initialize learning rate and momentum
  optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum=0.9)

  # load the data into a class that creates the batches
  data_loader = DataloaderFlowerSet.test_loader
  # Initialize model weights
  model.apply(weights_init)

  # Get a batch
  for i, data in enumerate(data_loader):
    # retrieve inputs and labels for this batch
    x_batch, y_batch = data
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward pass -- calculate model output
    pred = model(x_batch)
    # compute the loss
    loss = loss_function(pred, y_batch)
    # backward pass
    loss.backward()
    # SGD update
    optimizer.step()
    # Break out of this loop -- we just want to see the first
    # iteration, but usually we would continue
    break




model = Models.ResBlock(102)

from datetime import datetime

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
num_epochs = 2


train_losses = np.zeros(num_epochs)
val_losses = np.zeros(num_epochs)
train_accs = np.zeros(num_epochs)
val_accs = np.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train() 
    t0 = datetime.now()
    
    train_loss = []
    val_loss = []
    n_correct_train = 0
    n_total_train = 0

    
    for images, labels in DataloaderFlowerSet.train_loader:

        images = images.to(device)
        labels = labels.to(device)

        
        optimizer.zero_grad()
 
        y_pred = model(images)
        loss = criterion(y_pred, labels)  

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        # Compute training accuracy
        _, predicted_labels = torch.max(y_pred, 1)
        n_correct_train += (predicted_labels == labels).sum().item()
        n_total_train += labels.shape[0]

    train_loss = np.mean(train_loss)
    train_losses[epoch] = train_loss
    train_accs[epoch] = n_correct_train / n_total_train
    print(train_loss)
    # Validation phase
    model.eval()  
    n_correct_val = 0
    n_total_val = 0
    with torch.no_grad():  
        for images, labels in DataloaderFlowerSet.val_loader:
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            loss = criterion(y_pred, labels)

            # Store the validation loss
            val_loss.append(loss.item())

            # Compute validation accuracy
            _, predicted_labels = torch.max(y_pred, 1)
            n_correct_val += (predicted_labels == labels).sum().item()
            n_total_val += labels.shape[0]

    val_loss = np.mean(val_loss)
    val_losses[epoch] = val_loss
    val_accs[epoch] = n_correct_val / n_total_val
    duration = datetime.now() - t0

    # Print the metrics for the current epoch
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | '
          f'Duration: {duration}')

# Optionally, save the model after training
torch.save(model.state_dict(), "flower_classification_model.pth")