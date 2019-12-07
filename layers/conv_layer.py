from typing import Any

import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, drop_probability: float) -> None:
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.drop_probability = drop_probability

        self.torch_modules = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c_out),
            nn.ReLU(),
            nn.Dropout2d(self.drop_probability),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c_out),
            nn.ReLU(),
            nn.Dropout2d(self.drop_probability)
        )

    def forward(self, x: Any):
        return self.torch_modules(x)
