import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional
from torch import Tensor
from typing import List
from models.RTFVE import FeatureBlock
import models.basicblock as B


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


class ResNet_Discriminator(nn.Module):

    def __init__(self, nc, nb):
        super().__init__()
        unshuffle = B.conv(mode='F')
        head = B.conv(48, nc, mode='C')
        body = [B.ResBlock(nc, nc, mode='RCRC') for _ in range(nb)]
        tail = B.conv(nc, 1, mode='C')
        self.model = B.sequential(unshuffle, head, B.sequential(*body), tail)

    def forward(self, x):
        """
        Only the lr image. No ref images
        """
        # pass it through the model
        return self.model(x)
