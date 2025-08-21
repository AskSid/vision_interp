from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int = 1, padding: int = 0,
                 eps: float = 1e-3, momentum: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        ch_1x1: int,
        ch_3x3_reduce: int,
        ch_3x3: int,
        ch_5x5_reduce: int,
        ch_5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        # 1x1
        self.branch1 = BasicConv2d(in_ch, ch_1x1, kernel_size=1)

        # 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, ch_3x3_reduce, kernel_size=1),
            BasicConv2d(ch_3x3_reduce, ch_3x3, kernel_size=3, padding=1),
        )

        # 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            BasicConv2d(in_ch, ch_5x5_reduce, kernel_size=1),
            BasicConv2d(ch_5x5_reduce, ch_5x5, kernel_size=5, padding=2),
        )

        # 3x3 pool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_ch, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class TinyInceptionV1(nn.Module):
    """
    CIFAR-sized Tiny Inception-v1:
      Stem: BasicConv2d(3->32, 3x3, s=1, p=1)
      Group1 @ 32x32: A -> B -> MaxPool(3, s=2, p=0, ceil=True) -> 16x16
      Group2 @ 16x16: C -> D -> MaxPool(3, s=2, p=0, ceil=True) -> 8x8
      Group3 @  8x8: E -> F
      GAP -> Dropout -> FC(num_classes)
    """
    def __init__(self, num_classes: int = 100, dropout: float = 0.3) -> None:
        super().__init__()
        stem_channels = 32
        self.stem = BasicConv2d(3, stem_channels, kernel_size=3, stride=1, padding=1)

        # (1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj)
        A = (16, 16, 24, 4,  8,  8)   # out 56
        B = (16, 16, 24, 4,  8,  8)   # out 56
        C = (24, 24, 32, 6, 12, 18)   # out 86
        D = (24, 24, 40, 6, 12, 28)   # out 104
        E = (32, 32, 48, 8, 16, 40)   # out 136
        F = (32, 32, 56, 8, 16, 46)   # out 150

        blocks = []
        in_ch = stem_channels

        def add_block(spec):
            nonlocal in_ch, blocks
            c1, c3r, c3, c5r, c5, pp = spec
            blk = InceptionBlock(in_ch, c1, c3r, c3, c5r, c5, pp)
            blocks.append(blk)
            in_ch = c1 + c3 + c5 + pp

        # Group 1 (32x32)
        add_block(A)
        add_block(B)
        blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))  # -> 16x16

        # Group 2 (16x16)
        add_block(C)
        add_block(D)
        blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))  # -> 8x8

        # Group 3 (8x8)
        add_block(E)
        add_block(F)

        self.features = nn.Sequential(*blocks)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

class TinierInceptionV1(nn.Module):
    """
    CIFAR-sized Tinier Inception-v1 with ~half the parameters of TinyInceptionV1:
      Stem: BasicConv2d(3->22, 3x3, s=1, p=1)  
      Group1 @ 32x32: A -> B -> MaxPool(3, s=2, p=0, ceil=True) -> 16x16
      Group2 @ 16x16: C -> D -> MaxPool(3, s=2, p=0, ceil=True) -> 8x8
      Group3 @  8x8: E -> F
      GAP -> Dropout -> FC(num_classes)
    """
    def __init__(self, num_classes: int = 100, dropout: float = 0.3) -> None:
        super().__init__()
        stem_channels = 22
        self.stem = BasicConv2d(3, stem_channels, kernel_size=3, stride=1, padding=1)

        # (1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj)
        A = (11, 11, 17, 3, 6, 6)    # out 40 
        B = (11, 11, 17, 3, 6, 6)    # out 40 
        C = (17, 17, 23, 4, 8, 13)   # out 61 
        D = (17, 17, 28, 4, 8, 20)   # out 73 
        E = (23, 23, 34, 6, 11, 28)  # out 96 
        F = (23, 23, 40, 6, 11, 33)  # out 107 

        blocks = []
        in_ch = stem_channels

        def add_block(spec):
            nonlocal in_ch, blocks
            c1, c3r, c3, c5r, c5, pp = spec
            blk = InceptionBlock(in_ch, c1, c3r, c3, c5r, c5, pp)
            blocks.append(blk)
            in_ch = c1 + c3 + c5 + pp

        # Group 1 (32x32)
        add_block(A)
        add_block(B)
        blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))  # -> 16x16

        # Group 2 (16x16)
        add_block(C)
        add_block(D)
        blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))  # -> 8x8

        # Group 3 (8x8)
        add_block(E)
        add_block(F)

        self.features = nn.Sequential(*blocks)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)