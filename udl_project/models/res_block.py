import torch
import torch.nn as nn

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
