import torch.nn as nn

from project_spectrogram.module.stem_block import StemBlock
from project_spectrogram.module.res_block import ResBlock
from project_spectrogram.module.cbam import CBAM

class SpectrumExpertModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.stem = StemBlock(in_channels=3, out_channels=16)
        self.res1 = ResBlock(16, 32, stride=2)
        self.cbam1 = CBAM(32)
        self.res2 = ResBlock(32, 64, stride=2)
        self.cbam2 = CBAM(64)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 4. Bộ phân loại kết hợp
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Backbone
        x = self.stem(x)  # [B, 64, H, W]
        x = self.res1(x)  # [B, 128, H/2, W/2]
        x = self.cbam1(x)
        x = self.res2(x)  # [B, 256, H/4, W/4]
        x = self.cbam2(x)
        x = self.gap(x).flatten(1)  # [B, 256]
        return self.classifier(x)
