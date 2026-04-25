import math
import torch
import torch.nn as nn

class GaborFilterBank(nn.Module):
    """
    Tạo ngân hàng bộ lọc Gabor 2D cố định cho ảnh phổ.
    Args:
        in_channels: số kênh đầu vào
        orientations: số lượng hướng (4 bao gồm 0, 45, 90, 135 độ)
        kernel_size: kích thước kernel 21x21
        sigma: độ lệch chuẩn của Gaussian envelope
        frequency: tần số không gian của sóng sin
    """

    def __init__(self, in_channels=3, orientations=4, kernel_size=21, sigma=4.0, frequency=0.2):
        super().__init__()
        self.orientations = orientations
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.frequency = frequency

        # Tạo danh sách kernel
        kernels = []
        for theta in range(orientations):
            theta_rad = theta * (math.pi / orientations)  # 0, pi/4, pi/2, 3pi/4
            kernel = self._gabor_kernel(kernel_size, sigma, theta_rad, frequency)
            kernels.append(kernel)

        # Stack kernels: [out_channels, in_channels, H, W]
        weight = torch.stack(kernels).unsqueeze(1)  # [orientations, 1, H, W]
        weight = weight.repeat(1, in_channels, 1, 1)  # [orientations, in_channels, H, W]

        # Khởi tạo Conv2D từ kernel Gabor, sau đó cho phép học toàn bộ trọng số
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orientations,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.conv.weight = nn.Parameter(weight)

    def _gabor_kernel(self, size, sigma, theta, freq):
        """Tạo kernel Gabor 2D."""
        half = size // 2
        y, x = torch.meshgrid(torch.arange(-half, half + 1), torch.arange(-half, half + 1), indexing='ij')
        x = x.float()
        y = y.float()

        # Xoay tọa độ
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)

        # Gaussian envelope
        gauss = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))

        # Sóng sin phức (có thể lấy phần thực và phần ảo, hoặc chỉ phần thực)
        # Ở đây dùng phần thực
        sinusoid = torch.cos(2 * math.pi * freq * x_theta)

        kernel = gauss * sinusoid
        # Chuẩn hóa để tổng = 0 (tránh bias DC)
        kernel = kernel - kernel.mean()
        return kernel

    def forward(self, x):
        return self.conv(x)

