import torch
import torch.nn as nn


# Resnet from here: https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
# Bottleneck Block
class ResNetCNN(torch.nn.Module):
    def __init__(self, numClasses):
        super(ResNetCNN, self).__init__()
        self.conv1 = (nn.Conv2d(3, 16, kernel_size=3, padding=1),)

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
