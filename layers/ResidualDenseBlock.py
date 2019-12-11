import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class RDB(nn.Module):
    # creates an RDB of length 3
    def __init__(self, channels, depth=3, growth_rate=1, mask_size=3):
        super().__init__()

        self.depth = depth
        self.dense_layers = []
        self.channels = channels

        self.n_channels_ = channels
        for _ in range(0, depth):
            self.dense_layers.append(
                nn.Conv2d(in_channels=self.n_channels_, out_channels=growth_rate, kernel_size=mask_size,
                          padding=1,
                          bias=True).cuda())
            self.n_channels_ += growth_rate

        self.conv1x1 = nn.Conv2d(in_channels=self.n_channels_, out_channels=channels, kernel_size=1, stride=1, bias=False).cuda()

    def forward(self, x):
        self.previous = x
        for i, l in enumerate(self.dense_layers):
            self.intermediate = F.relu(l.forward(self.previous))
            self.previous = torch.cat((self.previous, self.intermediate), 1)
        self.intermediate = self.conv1x1.forward(self.previous)
        self.out = self.intermediate + x
        return self.out


# class make_dense(nn.Module):
#   def __init__(self, nChannels, growthRate, kernel_size=3):
#     super(make_dense, self).__init__()
#     self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
#   def forward(self, x):
#     self.intermediate = F.relu(self.conv(x))
#     self.out = torch.cat((x, self.intermediate), 1)
#     return self.out
#
# # Residual dense block (RDB) architecture
# class RDB(nn.Module):
#   def __init__(self, nChannels, nDenselayer, growthRate):
#     super(RDB, self).__init__()
#     nChannels_ = nChannels
#     modules = []
#     for i in range(nDenselayer):
#         modules.append(make_dense(nChannels_, growthRate))
#         nChannels_ += growthRate
#     self.dense_layers = nn.Sequential(*modules)
#     self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
#   def forward(self, x):
#     self.out = self.dense_layers(x)
#     self.out = self.conv_1x1(self.out)
#     self.out = self.out + x
#     return self.out


# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, nChannels, nDenseLayers=5, nFeatures=40, growthRate=16, nDenseBlocks=5, conv_layers=6):
        super(RDN, self).__init__()
        nChannel = nChannels
        nDenselayer = nDenseLayers

        growthRate = growthRate

        self.conv_layers_down = []
        self.dense_blocks = []
        self.conv_layers_up = []

        # F-1
        self.conv_layers_down.append(nn.Conv2d(nChannel, nFeatures, kernel_size=3, padding=1, bias=True).cuda())

        for _ in range(conv_layers - 1):
            self.conv_layers_down.append(nn.Conv2d(nFeatures, nFeatures, kernel_size=3, padding=1, bias=True).cuda())

        for _ in range(nDenseBlocks):
            self.dense_blocks.append(RDB(nFeatures, nDenselayer, growthRate))

        # global feature fusion (GFF)
        self.fusion_layer = nn.Conv2d(nFeatures * nDenseBlocks, nFeatures, kernel_size=1, padding=0, bias=True)

        for _ in range(conv_layers - 1):
            self.conv_layers_up.append(nn.Conv2d(nFeatures, nFeatures, kernel_size=3, padding=1, bias=True).cuda())

        self.conv_layers_up.append(nn.Conv2d(nFeatures, nChannel, kernel_size=3, padding=1, bias=True).cuda())

    def forward(self, x):

        # NOTE: Checkpointing is required otherwise you run out of memory FAST.


        # Convolutional layers down
        self.intermediate = x
        conv_outputs = []
        for l in self.conv_layers_down[:-1]:
            self.intermediate = checkpoint(l,self.intermediate)
            conv_outputs.append(self.intermediate)

        self.dense_input = checkpoint(self.conv_layers_down[-1],self.intermediate)

        # Dense block section
        self.intermediate = self.dense_input
        dense_outputs = []
        for l in self.dense_blocks:
            # checkpoints remove gradient info and recompute it during the backward pass to save memory
            self.intermediate = checkpoint(l,self.intermediate)
            dense_outputs.append(self.intermediate)

        # Fusion Layer
        self.fusion_input = torch.cat(dense_outputs, 1)
        self.fusion_output = checkpoint(self.fusion_layer,self.fusion_input)

        # Convolutional Up Layers
        self.intermediate = self.fusion_output
        for layer, output in zip(self.conv_layers_up[:-1], conv_outputs):
            self.intermediate = checkpoint(layer,self.intermediate)
            self.intermediate = self.intermediate + output

        self.intermediate = checkpoint(self.conv_layers_up[-1],self.intermediate)
        self.output = self.intermediate + x

        return self.output
