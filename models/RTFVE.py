import torch
import torch.nn as nn
from torchvision.models.shufflenetv2 import InvertedResidual
import torch.nn.functional as F
from typing import Any, Callable, List, Optional
from torch import Tensor
from typing import List
from torch.nn.utils import spectral_norm as norm


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, height, width)
    return x

# removed the batch norm from the shufflenetv2 source code in pytorch
class ShuffleBlock(nn.Module):

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleBlockWithSpectralNorm(nn.Module):

    def __init__(self, nc):
        super().__init__()
        bf = nc // 2
        self.conv0 = norm(nn.Conv2d(bf, bf, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv1 = norm(nn.Conv2d(bf, bf, kernel_size=3, stride=1, padding=1, groups=bf, bias=False))
        self.conv2 = norm(nn.Conv2d(bf, bf, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x: Tensor):
        x1, x2 = x.chunk(2, dim=1)
        x2 = F.relu(self.conv0(x2), inplace=True)
        x2 = F.relu(self.conv1(x2), inplace=True)
        x2 = F.relu(self.conv2(x2), inplace=True)
        out = torch.cat((x1, x2), dim=1)
        out = channel_shuffle(out, 2)
        return out



class FeatureBlock(nn.Module):
    """
    Create features for the lq input frames.
    It takes in the lq/hq frame and outputs features.
    The design: Conv + ResBlock * nb + Conv
    """

    def __init__(self, in_nc=3, out_nc=3, nc=32, nb=3, norm=False) -> None:
        super(FeatureBlock, self).__init__()
        self.head = nn.Conv2d(in_nc, nc, kernel_size=3, stride=1, padding=1)
        if norm:
            self.body = nn.Sequential(*[ShuffleBlockWithSpectralNorm(nc) for _ in range(nb)])
        else:
            self.body = nn.Sequential(*[ShuffleBlock(nc, nc, 1) for _ in range(nb)])
        self.tail = nn.Conv2d(nc, out_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class SpatialTransformerAlign(nn.Module):
    """
    This block aligns the reference and lr features together
    using a spatial transformer.
    """

    def __init__(self, lr) -> None:
        super().__init__()

        # the localization network designed for 64x64 input
        if lr == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2), # 64
                nn.ReLU(True),
                ShuffleBlock(32, 32, 1),
                ShuffleBlock(32, 32, 2),   # 32
                ShuffleBlock(32, 32, 1),
                ShuffleBlock(32, 32, 2),    # 16
                ShuffleBlock(32, 32, 1),
                ShuffleBlock(32, 32, 2),    # 8
                ShuffleBlock(32, 32, 1),
                ShuffleBlock(32, 32, 2),    # 4
                ShuffleBlock(32, 32, 1),
            )
        else:
            raise ValueError('lr for SpatialTransformerAlign should be 64. Modify the code' \
            'for other latent spatial sizes using the Shuffleblock.')
        
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, lq, ref):
        """
        lq: lq features
        ref: ref features
        """
        # concatenate lq and rfd
        x = torch.cat((lq, ref), 1)

        # pass it through the conv layers
        x = self.conv(x)

        # reshape it so that it can be passed to the fc layers
        x = x.view(-1, 4 * 4 * 32)
        theta = self.fc(x)
        
        # perform the sampling of rf (not rfd) based on the theta parameters
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, ref.size(), align_corners=False)
        ref_a = F.grid_sample(ref, grid, align_corners=False, padding_mode='reflection')

        return ref_a

class FeatureAggregationExp(nn.Module):
    """
    Performs the feature aggregation step so that information is taken in
    a weighted fashion from all the aligned references.

    This uses softmax and a modification of the frobenius norm.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, lq_f: Tensor, refs_a: List[Tensor]):
        # get count of refs
        rcount = len(refs_a)
        # get the modified norm distance for the pixel positions
        refs_sub = [lq_f.sub(r).square().sum(dim=1, keepdim=True).sqrt() 
                    for r in refs_a]
        # concatenate the above tensors and clamp to avoid div by 0
        refs_sub = torch.cat(refs_sub, 1).clamp(max=1e2)
        # get the softmax of the negative to prioritize smaller distance from lq
        smax = (refs_sub.neg().exp()) / (refs_sub.neg().exp().sum(dim=1, keepdim=True) + 1e-9)
        # concatenate the given refs
        refs_a = torch.cat(refs_a, 1)
        # multiply the softmax weights to the provided refs
        refs_a = refs_a.mul(smax.repeat_interleave(rcount, dim=1))
        # aggregate all the weigthed refs
        # refs_a = sum(refs_a.tensor_split(rcount, dim=1))
        refs_a = refs_a.tensor_split(rcount, dim=1)
        tot = torch.zeros_like(refs_a[0])
        for t in refs_a:
            tot += t
        return tot + lq_f
        # return refs_a + lq_f
        # return torch.cat([lq_f, refs_a], 1)

"""
Main model
"""

class RTFVE(nn.Module):

    def __init__(self, nc, uf, lr) -> None:
        super(RTFVE, self).__init__()

        # pixelunshuffle for the ref images
        self.unshuffle = nn.PixelUnshuffle(uf)
        
        # the feature extractors
        self.feature_extraction_lq = FeatureBlock(in_nc=int(3*uf**2), nb=10, nc=nc)
        self.feature_extraction_ref = FeatureBlock(in_nc=int(3*uf**2), nb=10, nc=nc)   

        # the feature alignment block
        self.feature_align = SpatialTransformerAlign(lr)

        # the feature aggregation block
        self.feature_aggregation = FeatureAggregationExp()

        # the reconstruction block
        self.reconstruction = FeatureBlock(out_nc=int(3*uf**2), nc=nc, nb=10)

        # pixel shuffle
        self.shuffle = nn.PixelShuffle(uf)

        # last convolution
        self.last = nn.Conv2d(3, 3, 3, padding=1)

    def extract_ref_features(self, refs: List[Tensor]) -> List[Tensor]:
        refs_f = [self.unshuffle(r) for r in refs]
        refs_f = [self.feature_extraction_ref(r) for r in refs_f]
        return refs_f

    def forward(self, lq: Tensor, refs: List[Tensor], features: bool = False):
        """
        lq: The low quality frame.
        refs: The high quality reference frames in a list.
        """
        # unshuffle and extract features for lq image
        lq_f = self.unshuffle(lq)
        lq_f = self.feature_extraction_lq(lq_f)
        

        # unshuffle and extract features for rf images if not provided
        if not features:
            refs_f = [self.unshuffle(r) for r in refs]
            refs_f = [self.feature_extraction_ref(r) for r in refs_f]
        else:
            refs_f = refs

        # align ref features
        refs_f = [self.feature_align(lq_f, r) for r in refs_f]

        # aggregate all features
        feat_agg = self.feature_aggregation(lq_f, refs_f)

        # produce the final output
        pred = self.reconstruction(feat_agg)

        # get back to hq size
        pred = self.shuffle(pred)
        pred = self.last(pred)

        return (lq + pred).clamp(min=0, max=1)
