import torch
import torch.nn as nn

class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Nhánh thời gian (1x7)
        self.temporal_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 7), padding=(0, 3))
        # Nhánh tần số (7x1)
        self.freq_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=(7, 1), padding=(3, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.temporal_conv(x)
        x2 = self.freq_conv(x)
        x = torch.cat([x1, x2], dim=1)
        return self.relu(self.bn(x))