import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self, numClasses):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(inplace=True), nn.Linear(128, numClasses)
        )

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


# Resnet from here: https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
# Bottleneck Block


# overfits as fuck
class ResBlock(torch.nn.Module):
    def __init__(self, numClasses):
        super().__init__()
        in_channels = 3
        out_channels = 16

        self.bottlenck1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.bottlenck2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential()

        self.classifier = nn.Sequential(nn.Linear(3072, numClasses))

    def forward(self, x):
        out = self.bottlenck1(x)
        out += self.shortcut1(x)
        out = self.bottlenck2(out)
        out += self.shortcut2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
