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


class DeepConvNet(nn.Module):
    def __init__(self, nCh=22, nTime=1000, cls=4, dropout=0.25) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        kernel_size = [1, 10]
        filter_layer = [25, 50, 100, 200]

        first_layer = nn.Sequential(
            Conv2dWithNorm(1, 25, kernel_size, max_norm=2),
            Conv2dWithNorm(25, 25, (nCh, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3))
        )
        middle_layer = nn.Sequential(*[
            nn.Sequential(
                nn.Dropout(dropout),
                Conv2dWithNorm(in_f, out_f, kernel_size),
                nn.BatchNorm2d(out_f),
                nn.ELU(),
                nn.MaxPool2d((1, 3))
            ) for in_f, out_f in zip(filter_layer, filter_layer[1:])
        ])
        self.conv_layer = nn.Sequential(first_layer, middle_layer)

        linear_in = self._forward_flatten().shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            LinearWithNorm(linear_in, cls, max_norm=0.5),
            nn.LogSoftmax(dim=1)
        )

    def _forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        x = self.conv_layer(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return x

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    net = DeepConvNet(22, 1000).cuda()
    summary(net, (64, 1, 22, 1000))