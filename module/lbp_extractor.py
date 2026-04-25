import torch
import torch.nn as nn
import torch.nn.functional as F

class LBPExtractor(nn.Module):
    def __init__(self, threshold_scale=10.0):
        super().__init__()
        # Gộp 8 kernel vào 1 tensor [8, 1, 3, 3]
        # Thứ tự khớp với weights: R, L, BC, BL, TC, TR, BR, TL
        kernels = torch.tensor([
            [[[1, 0, 0],
              [0, -1, 0],
              [0, 0, 0]]],  # top-left

            [[[0, 1, 0],
              [0, -1, 0],
              [0, 0, 0]]],  # top

            [[[0, 0, 1],
              [0, -1, 0],
              [0, 0, 0]]],  # top-right

            [[[0, 0, 0],
              [0, -1, 1],
              [0, 0, 0]]],  # right

            [[[0, 0, 0],
              [0, -1, 0],
              [0, 0, 1]]],  # bottom-right

            [[[0, 0, 0],
              [0, -1, 0],
              [0, 1, 0]]],  # bottom

            [[[0, 0, 0],
              [0, -1, 0],
              [1, 0, 0]]],  # bottom-left

            [[[0, 0, 0],
              [1, -1, 0],
              [0, 0, 0]]]  # left
        ], dtype=torch.float32)
        kernels = kernels.repeat(1, 3, 1, 1)
        self.kernels = nn.Parameter(kernels)

        # Trọng số bit trainable để mô-đun có thể tự điều chỉnh đóng góp từng hướng
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32).view(1, 8, 1, 1)
        self.weights = nn.Parameter(weights)
        self.threshold_scale = nn.Parameter(torch.tensor(float(threshold_scale)))

    def forward(self, x):
        # x: [B, 3, H, W] -> diff: [B, 8, H, W]
        diff = F.conv2d(x, self.kernels, padding=1)

        # Dùng ngưỡng mềm để giữ gradient đi qua extractor
        scale = self.threshold_scale.clamp(min=1e-3, max=100.0)
        bit_mask = torch.sigmoid(diff * scale) * self.weights

        # Ép dọc theo channel dim -> [B, 1, H, W]
        lbp_map = bit_mask.sum(dim=1, keepdim=True)
        return lbp_map / 255.0
