import torch
import torch.nn as nn
import torch.nn.functional as F

class DoGDetector(nn.Module):
    """
    Xấp xỉ Laplacian of Gaussian bằng Difference of Gaussians.
    """

    def __init__(self, sigma1=1.0, sigma2=2.0, kernel_size=9):
        super().__init__()
        self.raw_sigma1 = nn.Parameter(torch.tensor(float(sigma1)))
        self.raw_sigma2 = nn.Parameter(torch.tensor(float(sigma2)))
        self.kernel_size = kernel_size

        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def _gaussian_kernel(self, sigma):
        sigma = F.softplus(sigma) + 1e-4
        kernel = torch.exp(-(self.x ** 2 + self.y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        return kernel

    def forward(self, x):
        channels = x.shape[1]
        g1 = self._gaussian_kernel(self.raw_sigma1).to(dtype=x.dtype)
        g2 = self._gaussian_kernel(self.raw_sigma2).to(dtype=x.dtype)
        g1 = g1.view(1, 1, self.kernel_size, self.kernel_size).repeat(1, channels, 1, 1)
        g2 = g2.view(1, 1, self.kernel_size, self.kernel_size).repeat(1, channels, 1, 1)

        # Áp dụng hai Gaussian và lấy hiệu
        out1 = F.conv2d(x, g1, padding=self.kernel_size // 2)
        out2 = F.conv2d(x, g2, padding=self.kernel_size // 2)
        dog = out1 - out2

        # Chuẩn hóa về [0,1]
        dog_min = dog.amin(dim=(1, 2, 3), keepdim=True)
        dog_max = dog.amax(dim=(1, 2, 3), keepdim=True)
        dog = (dog - dog_min) / (dog_max - dog_min + 1e-8)
        return dog
