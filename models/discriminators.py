from torch import Tensor
import torch.nn as nn
from torch import Tensor
from models.RTFVE import FeatureBlock


class Shuffle_Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(4)
        self.model = FeatureBlock(in_nc=48, out_nc=1, nc=64, nb=10, norm=False)

    def forward(self, x: Tensor):
        return self.model(self.unshuffle(x))


class Shuffle_Discriminator_Norm(nn.Module):

    def __init__(self):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(4)
        self.model = FeatureBlock(in_nc=48, out_nc=1, nc=64, nb=10, norm=True)

    def forward(self, x: Tensor):
        return self.model(self.unshuffle(x))
