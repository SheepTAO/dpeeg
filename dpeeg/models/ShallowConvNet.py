'''
References
----------
R. T. Schirrmeister et al., “Deep learning with convolutional neural networks 
for EEG decoding and visualization,” Human Brain Mapping, vol. 38, no. 11, pp.
5391-5420, 2017, doi: 10.1002/hbm.23730.
'''


import torch
import torch.nn as nn

from .utils import Conv2dWithNorm, LinearWithNorm


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ShallowConvNet(nn.Module):
    def __init__(self, nCh=22, nTime=1000, cls=4, F=40, C=14, P=35, S=7, 
                 dropout=0.5) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime

        self.conv = nn.Sequential(
            Conv2dWithNorm(1, F, (1, 14), max_norm=2, bias=False),
            Conv2dWithNorm(F, F, (nCh, 1), max_norm=2, bias=False, groups=F),
            nn.BatchNorm2d(F),
            Lambda(torch.square),
            nn.AvgPool2d((1, P), stride=(1, S)),
            Lambda(torch.log),
        )

        linear_in = self.forward_flatten().shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            LinearWithNorm(linear_in, cls, max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

    def forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        x = self.conv(x)
        x = torch.flatten(x, 1, -1)
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    net = ShallowConvNet().cuda()
    summary(net, (32, 1, 22, 1000))
    