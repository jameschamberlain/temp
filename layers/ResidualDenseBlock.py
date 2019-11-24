import torch.nn as nn
import torch.nn.functional as F
import torch


class RDB(nn.Module):
    # creates an RDB of length 3
    def __init__(self, channels, depth=3,  mask_size=3, growth_rate = 1):
        super().__init__()

        self.depth=depth
        self.dense_layers = []
        self.merge_layers = []

        intermediate_channels = channels
        for _ in range(0,depth):
            self.dense_layers.append(nn.Conv2d(in_channels=intermediate_channels,out_channels=growth_rate,kernel_size=mask_size,bias=True))
            self.intermediate_channels += channels

        self.conv1x1 = nn.Conv2d(in_channels=self.depth, out_channels=growth_rate, kernel_size=1, stride=1, bias=False)


    def forward(self,input:torch.Tensor):
        l: nn.Conv2d
        previous = input
        for i,l in enumerate(self.conv_layers):
            intermediate = F.relu(l.forward(previous))
            previous = torch.cat((previous,intermediate), 1)
        out = previous + input
        return out





