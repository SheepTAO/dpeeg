'''
EEGNet-8,2

References
----------
V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance,
“EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,”
J. Neural Eng., vol. 15, no. 5, p. 056013, Jul. 2018, doi: 10.1088/1741-2552/aace8c.
'''


import torch
import torch.nn as nn


class Conv2dWithNorm(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, 2, 0, self.max_norm
            )
        return super().forward(input)


class LinearWithNorm(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, 2, 0, self.max_norm
            )
        return super().forward(input)


class EEGNet(nn.Module):
    def __init__(self, nCh=22, nTime=1000, C1=125, F1=8, D=2, F2=16, C2=15, 
                 P1=4, P2=8, p=0.25, cls=4) -> None:
        super().__init__()

        self.filter = nn.Sequential(
            nn.Conv2d(1, F1, (1, C1), padding=(0, C1//2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwise_conv = nn.Sequential(
            Conv2dWithNorm(F1, D*F1, (nCh, 1), groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(p)
        )

        self.separable_conv = nn.Sequential(
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
        out = self.depthwise_conv(out)
        out = self.separable_conv(out)
        return self.flatten(out).size(1)

    def forward(self, x):
        out = self.filter(x)
        out = self.depthwise_conv(out)
        out = self.separable_conv(out)
        out = self.flatten(out)
        return self.fc(out)


if __name__ == '__main__':
    from torchinfo import summary
    net = EEGNet(22, 1000).cuda()
    summary(net, (1, 1, 22, 1000))