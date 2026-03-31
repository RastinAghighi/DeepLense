"""EDSR-baseline model for single-image super-resolution.

Reference: Lim et al., "Enhanced Deep Residual Networks for Single Image
Super-Resolution", CVPRW 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block without batch normalization.

    Batch normalization is omitted because it normalises feature magnitudes,
    which limits the range flexibility needed for pixel-regression tasks like
    super-resolution.  Removing BN also saves GPU memory and allows larger
    models or batch sizes.

    A small residual scaling factor (default 0.1) multiplies the residual
    branch before the addition.  This prevents gradient explosion when stacking
    many residual blocks and stabilises early training.
    """

    def __init__(self, n_feats: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))


class EDSR(nn.Module):
    """EDSR-baseline: 16 residual blocks, 64 filters, 2x upscale, no BN.

    The network learns only the high-frequency residual between the
    bicubic-upsampled input and the ground-truth HR image (global residual
    learning).  This lets the model focus its capacity on recovering fine
    detail rather than reproducing low-frequency content that a simple
    bicubic interpolation already handles well.
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_feats: int = 64,
        n_resblocks: int = 16,
        scale: int = 2,
    ) -> None:
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1)

        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, n_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bicubic = F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        head = self.head(x)
        body = self.body(head)
        res = head + body  # long skip connection
        sr = self.tail(res)
        return sr + bicubic  # global residual


if __name__ == "__main__":
    model = EDSR()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EDSR-baseline  |  parameters: {n_params:,}")
