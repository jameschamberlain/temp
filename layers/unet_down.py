import torch.nn as nn
import hyper_param
from layers.conv_layer import ConvLayer

from torch.nn import functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class UNetDown(nn.Module):
    def __init__(self, c_in: int, c_out: int, c: int, image_size, n_fc_layers = 10) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c = c
        self.n_pool_layers = hyper_param.POOL_LAYERS
        self.drop_prob = hyper_param.DROPOUT
        img_size = image_size

        self.down_sample_layers = nn.ModuleList([ConvLayer(self.c_in, self.c, self.drop_prob)])
        img_size = image_size // 4
        channels = self.c

        for _ in range(self.n_pool_layers - 1):
            img_size = image_size // 4
            self.down_sample_layers += [ConvLayer(channels, channels * 2, self.drop_prob)]
            channels *= 2


        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(self.drop_prob)
        )

        fc_layers = [Flatten()]
        for _ in range(n_fc_layers):
                fc_layers.append(nn.Linear(img_size*channels,img_size*channels))
                fc_layers.append(nn.Tanh())
        fc_layers.append(nn.Linear(img_size*channels,1))
        fc_layers.append(nn.Sigmoid())

        self.fc_layers = nn.Sequential(*fc_layers)



    def forward(self, x: Any, ):
        stack = []
        y = x
        for layer in self.down_sample_layers:
            y = layer(y)
            stack.append(y)
            y = F.max_pool2d(y, kernel_size=2)
        y = self.conv(y)
        y = self.fc_layers.forward(y)
        return y
