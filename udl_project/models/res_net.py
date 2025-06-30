import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    A Bottleneck block as used in ResNet architectures.

    Bottleneck blocks are designed to make very deep neural networks more efficient and easier to train by reducing the number of parameters (in channels > bottleneck_channels) and computational cost. They achieve this by using a three-layer structure:
        1. A 1x1 convolution reduces the number of channels (dimensionality reduction).
        2. A 3x3 convolution processes the reduced representation.
        3. A 1x1 convolution restores the original number of channels (dimensionality expansion).

    Each convolution is followed by batch normalization and a ReLU activation. The input is added to the output (skip connection), enabling better gradient flow and mitigating the vanishing gradient problem in deep networks.

    Args:
        in_channels (int): Number of input channels.
        _bottleneck_channels (int): Number of channels in the bottleneck (intermediate) layers.
        out_channels (int): Number of output channels.

    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H, W)
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        super(Bottleneck, self).__init__()
        stride = 2 if False else 1

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if False or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor):
        # x -> torch.Size([32, 3, 64, 64])
        out = self.conv1(self.relu(self.bn1(x)))
        # out -> torch.Size([32, 16, 64, 64])
        out = self.conv2(self.relu(self.bn2(out)))
        # out -> torch.Size([32, 16, 64, 64])
        out = self.conv3(self.relu(self.bn3(out)))
        # out -> torch.Size([32, 3, 64, 64])
        out += self.shortcut(x)
        # out ->
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet, self).__init__()

        in_channels = 3

        # Initial feature extraction: Conv7x7 -> BN -> ReLU -> MaxPool
        self.pre = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # First bottleneck block: input 64 channels, bottleneck 16, output 64
        self.bottleneck_one = Bottleneck(in_channels=64, bottleneck_channels=16, out_channels=64)

        # Second bottleneck block: input 64 channels, bottleneck 32, output 128
        self.bottleneck_two = Bottleneck(
            in_channels=64,
            bottleneck_channels=32,
            out_channels=128,
        )

        # Third bottleneck block: input 128 channels, bottleneck 64, output 256
        self.bottleneck_three = Bottleneck(
            in_channels=128,
            bottleneck_channels=64,
            out_channels=256,
        )

        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer for classification
        self.classifier = nn.Linear(256 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor):
        # torch.Size([32, 3, 64, 64])
        # x: [batch_size, channels (RGB=3), height=64, width=64]

        out = self.pre(x)
        # torch.Size([32, 64, 32, 32])

        out = self.bottleneck_one(out)
        out = self.bottleneck_two(out)
        out = self.bottleneck_three(out)

        out = self.global_pool(out)
        out = torch.flatten(out, start_dim=1)

        return self.classifier(out)
