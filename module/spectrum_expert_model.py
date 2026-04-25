import torch
import torch.nn as nn

from project_spectrogram.module.gabor_filter_bank import GaborFilterBank
from project_spectrogram.module.dog_detector import DoGDetector
from project_spectrogram.module.lbp_extractor import LBPExtractor
from project_spectrogram.module.metadata_embedding import MetadataEmbedding
from project_spectrogram.module.stem_block import StemBlock
from project_spectrogram.module.res_block import ResBlock
from project_spectrogram.module.cbam import CBAM

class SpectrumExpertModel(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # 1. Bộ lọc cổ điển
        self.gabor = GaborFilterBank(in_channels=3, orientations=4, kernel_size=21)
        self.dog = DoGDetector(sigma1=1.0, sigma2=2.0, kernel_size=9)
        self.lbp = LBPExtractor()

        # 2. Nhúng siêu dữ liệu
        self.meta_embed = MetadataEmbedding(input_dim=2, embedding_dim=32)

        # 3. Backbone CNN
        # Đầu vào sẽ có 4 kênh: ảnh gốc(3) + gabor(4) + dog(1) + lbp(1) = 9
        # Chú ý: Gabor có 4 kênh đầu ra, tổng cộng: 3 + 4 + 1 + 1 = 9 kênh.
        # Phép chiếu 9 channels -> 8 channels để tối ưu tensor core
        self.channel_projection = nn.Conv2d(9, 8, kernel_size=1)
        self.stem = StemBlock(in_channels=8, out_channels=64)
        self.res1 = ResBlock(64, 128, stride=2)
        self.cbam1 = CBAM(128)
        self.res2 = ResBlock(128, 256, stride=2)
        self.cbam2 = CBAM(256)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 4. Bộ phân loại kết hợp
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_spect, x_meta = None):
        # x_spect: [B, 3, H, W]
        # x_meta:  [B, 2]

        # Trích xuất đặc trưng thủ công
        gabor_feat = self.gabor(x_spect)  # [B, 4, H, W]
        dog_feat = self.dog(x_spect)  # [B, 1, H, W]
        lbp_feat = self.lbp(x_spect)  # [B, 1, H, W]

        # Ghép kênh đầu vào
        x_rich = torch.cat([x_spect, gabor_feat, dog_feat, lbp_feat], dim=1)  # [B, 9, H, W]

        # Backbone
        x = self.channel_projection(x_rich)
        x = self.stem(x)  # [B, 64, H, W]
        x = self.res1(x)  # [B, 128, H/2, W/2]
        x = self.cbam1(x)
        x = self.res2(x)  # [B, 256, H/4, W/4]
        x = self.cbam2(x)

        x = self.gap(x).flatten(1)  # [B, 256]

        if x_meta is None:
            x_meta = torch.zeros(x_spect.size(0), 2, device=x_spect.device, dtype=x_spect.dtype)

        meta_emb = self.meta_embed(x_meta) # [B, 32]
        x = torch.cat([x, meta_emb], dim=1) # [B, 288]

        return self.classifier(x)
