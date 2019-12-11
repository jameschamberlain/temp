import torch.nn as nn
import torch.nn.functional as F
import torch

class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)
    def forward(self, x):
        x = self.body(x)
        return x


# class RDB(nn.Module):
#     # creates an RDB of length 3
#     def __init__(self, channels, depth=3,  mask_size=3, growth_rate = 1):
#         super().__init__()
#
#         self.depth=depth
#         self.dense_layers = []
#
#         intermediate_channels = channels
#         for _ in range(0,depth):
#             self.dense_layers.append(nn.Conv2d(in_channels=intermediate_channels,out_channels=growth_rate,kernel_size=mask_size,bias=True))
#             intermediate_channels += channels
#
#         self.conv1x1 = nn.Conv2d(in_channels=self.depth, out_channels=growth_rate, kernel_size=1, stride=1, bias=False)
#
#
#     def forward(self,input:torch.Tensor):
#         l: nn.Conv2d
#         previous = input
#         previous
#         for i,l in enumerate(self.dense_layers):
#             intermediate = F.relu(l.forward(previous))
#             previous = torch.cat((previous,intermediate), 1)
#         out = previous + input
#         return out

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, nChannels, nDenseLayers=4, nFeatures=64, growthRate=32, nDenseBlocks=20, conv_layers=6):
        super(RDN, self).__init__()
        nChannel = nChannels
        nDenselayer = nDenseLayers

        growthRate = growthRate

        self.conv_layers_down = []
        self.dense_blocks = []
        self.conv_layers_up = []

        # F-1
        self.conv_layers_down.append(nn.Conv2d(nChannel, nFeatures, kernel_size=3, padding=1, bias=True))

        for _ in range(conv_layers - 1):
            self.conv_layers_down.append(nn.Conv2d(nFeatures, nFeatures, kernel_size=3, padding=1, bias=True))

        for _ in range(nDenseBlocks):
            self.dense_blocks.append(RDB(nFeatures, nDenselayer, growthRate))

        # global feature fusion (GFF)
        self.fusion_layer = nn.Conv2d(nFeatures * nDenseBlocks, nFeatures, kernel_size=1, padding=0, bias=True)

        for _ in range(conv_layers - 1):
            self.conv_layers_up.append(nn.Conv2d(nFeatures, nFeatures, kernel_size=3, padding=1, bias=True))

        self.conv_layers_up.append(nn.Conv2d(nFeatures, nChannel, kernel_size=3, padding=1, bias=True))

    def forward(self, x):

        # Convolutional layers down
        intermediate = x.cuda()
        conv_outputs = []
        for l in self.conv_layers_down[:-1]:
            print("progressed a layer")
            intermediate = l.forward(intermediate)
            conv_outputs.append(intermediate)

        dense_input = self.conv_layers_down[-1].forward(intermediate)

        # Dense block section
        intermediate = dense_input
        dense_outputs = []
        for l in self.dense_blocks:
            intermediate = l.forward(intermediate)
            dense_outputs.append(intermediate)

        # Fusion Layer
        fusion_input = torch.cat(dense_outputs, 1)
        fusion_output = self.fusion_layer(fusion_input)

        # Convolutional Up Layers
        intermediate = fusion_output
        for layer,output in zip(self.conv_layers_up[:-1],conv_outputs):
            layer.forward(intermediate)
            intermediate = intermediate + output

        intermediate = self.conv_layers_up[-1].forward(intermediate)
        output = intermediate + x

        return output



