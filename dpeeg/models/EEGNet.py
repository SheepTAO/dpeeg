'''
    EEGNet-8,2
    https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
'''


import torch
from torch import Tensor, nn


class Conv2dWithNorm(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, maxNorm=1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxNorm = maxNorm
        self.doWeightNorm = doWeightNorm

    def forward(self, input: Tensor) -> Tensor:
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, 2, 0, self.maxNorm
            )
        return super().forward(input)


class LinearWithNorm(nn.Linear):
    def __init__(self, *args, maxNorm, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxNorm = maxNorm

    def forward(self, input: Tensor) -> Tensor:
        self.weight.data = torch.renorm(
            self.weight.data, 2, 0, self.maxNorm
        )
        return super().forward(input)


class EEGNet(nn.Module):
    def __init__(self, nCh, nTime, C1=63, F1=8, D=2, F2=16, C2=15, P1=8, P2=16,
                 p=0.5, cls=4) -> None:
        super().__init__()

        self.filter = nn.Sequential(
            nn.Conv2d(1, F1, (1, C1), padding=(0, C1//2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithNorm(F1, D*F1, (nCh, 1), groups=F1, bias=False, maxNorm=1,
                           doWeightNorm=False),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(p)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, C2), padding=(0, C2//2), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(p)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self.get_size(nCh, nTime), cls, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def get_size(self, nCh, nTime):
        x = torch.randn(1, 1, nCh, nTime)
        out = self.filter(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        return self.flatten(out).size(1)

    def forward(self, x):
        out = self.filter(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = self.flatten(out)
        return self.fc(out)
