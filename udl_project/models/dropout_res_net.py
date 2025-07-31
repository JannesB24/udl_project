import torch
from torch import nn
from udl_project.models.res_net import ResNet, Bottleneck


class DPBottleneck(Bottleneck):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride = 1,dp_rate= 0.5):
        super().__init__(in_channels, bottleneck_channels, out_channels, stride)
        self.dropout = nn.Dropout(p=dp_rate) 
   
    
    def forward(self, x: torch.Tensor):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.dropout(self.conv2(self.relu(self.bn2(out))))
        out = self.dropout(self.conv3(self.relu(self.bn3(out))))
        out += self.shortcut(x)
        return out

class DPResNet(ResNet):
    def __init__(self, num_classes,dp_rate):
        super().__init__(num_classes)
        in_channels = 3
        self.dropout = nn.Dropout(p=dp_rate) 
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
        self.bottleneck_one = DPBottleneck(in_channels=64, bottleneck_channels=16, out_channels=64,dp_rate=dp_rate)

        # Second bottleneck block: input 64 channels, bottleneck 32, output 128
        self.bottleneck_two = DPBottleneck(
            in_channels=64,
            bottleneck_channels=32,
            out_channels=128,
            dp_rate=dp_rate
        )

        # Third bottleneck block: input 128 channels, bottleneck 64, output 256
        self.bottleneck_three = DPBottleneck(
            in_channels=128,
            bottleneck_channels=64,
            out_channels=256,
            dp_rate=dp_rate
        )

        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer for classification
        # 256 * 8 * 8: 256 channels with 8x8 spatial dimensions after pooling
        self.classifier = nn.Linear(256 * 8 * 8, num_classes)
