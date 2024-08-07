'''
References
----------
R. Mane et al., “FBCNet: A Multi-view Convolutional Neural Network for Brain-
Computer Interface,” arXiv.org. Accessed: Mar. 09, 2023. [Online]. 
Available: https://arxiv.org/abs/2104.01233v1
'''


import torch
from torch import nn

from .utils import Conv2dWithNorm, LinearWithNorm


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6)
        return torch.log(x)


class FBCNet(nn.Module):
    def __init__(self, nCh, nTime, cls, bands=9, m=32, stride=4) -> None:
        super().__init__()
        self.stride = stride

        self.scb = nn.Sequential(
            Conv2dWithNorm(bands, m * bands, (nCh, 1), groups=bands, 
                           do_weight_norm=True, max_norm=2, padding=0),
            nn.BatchNorm2d(m * bands),
            swish(),
        )

        self.temporal_layer = LogVarLayer(dim=3)

        self.head = nn.Sequential(
            nn.Flatten(),
            LinearWithNorm(m*bands*stride, cls, do_weight_norm=True, max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.scb(x)
        x = x.reshape([*x.shape[:2], self.stride, x.shape[3]//self.stride])
        x = self.temporal_layer(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    net = FBCNet(22, 1000, 4).cuda()
    summary(net, (1, 9, 22, 1000))