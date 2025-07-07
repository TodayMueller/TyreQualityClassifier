import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.scale = nn.Parameter(torch.ones(num_features))  # Масштаб по каналам
        self.bias = nn.Parameter(torch.zeros(num_features))  # Смещение по каналам

    def forward(self, x):
        x = self.bn(x)
        # Расширяем scale и bias при необходимости
        if x.dim() == 2:
            return x * self.scale + self.bias
        elif x.dim() == 3:  # [B, C, T]
            return x * self.scale.view(1, -1, 1) + self.bias.view(1, -1, 1)
        else:
            raise ValueError(f"Unsupported input shape for AdaptiveBatchNorm1d: {x.shape}")

class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.scale = nn.Parameter(torch.ones(num_features, 1, 1))  # масштаб по каналам
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))  # смещение

    def forward(self, x):
        x = self.bn(x)
        return x * self.scale + self.bias

class MaxDepthPool2d(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        shape = x.shape
        channels = shape[1] // self.pool_size
        new_shape = (shape[0], channels, self.pool_size, *shape[-2:])
        return torch.amax(x.view(new_shape), dim=2)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=8):
        super().__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.feed_forward = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, squeeze_channels),
            nn.Mish(),
            nn.Linear(squeeze_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.feed_forward(x).view(-1, x.size(1), 1, 1)

class ResidualConnection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        squeeze_active=False,
        squeeze_factor=8,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.squeeze_active = squeeze_active
        self.squeeze_excitation = SqueezeExcitation(out_channels, squeeze_factor)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut_connection = nn.Sequential()
        if not in_channels == out_channels or stride > 1:
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x_residual = self.feed_forward(x)
        x_shortcut = self.shortcut_connection(x)
        residual_output = F.mish(x_residual + x_shortcut)
        if self.squeeze_active:
            return self.squeeze_excitation(residual_output) + x_shortcut
        return residual_output
    
# Важно: Строим модель через nn.Sequential, как в исходном обучении
ClassificationNet = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
    AdaptiveBatchNorm2d(num_features=32),
    nn.Mish(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualConnection(32, 64, kernel_size=3, stride=1, squeeze_active=True),
    ResidualConnection(64, 64, kernel_size=3, stride=1, squeeze_active=True),
    MaxDepthPool2d(pool_size=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualConnection(32, 96, kernel_size=5, stride=1, squeeze_active=True),
    ResidualConnection(96, 96, kernel_size=5, stride=1, squeeze_active=True),
    MaxDepthPool2d(pool_size=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ResidualConnection(48, 128, kernel_size=3, stride=1, squeeze_active=True),
    ResidualConnection(128, 128, kernel_size=3, stride=1, squeeze_active=True),
    MaxDepthPool2d(pool_size=4),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(32 * 10 * 10, 256, bias=False),
    AdaptiveBatchNorm1d(num_features=256),
    nn.Mish(),
    nn.Dropout(0.3),
    nn.Linear(256, 256, bias=False),
    AdaptiveBatchNorm1d(num_features=256),
    nn.Mish(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
)