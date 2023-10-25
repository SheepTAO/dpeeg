'''
References
----------
R. T. Schirrmeister et al., “Deep learning with convolutional neural networks 
for EEG decoding and visualization,” Human Brain Mapping, vol. 38, no. 11, pp.
5391–5420, 2017, doi: 10.1002/hbm.23730.
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


class ShallowNet(nn.Module):
    def __init__(self, nCh=22, nTime=1000, cls=4, F1=40) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Conv2dWithNorm(1, F1, (1, 25), max_norm=2, bias=False),
            Conv2dWithNorm(F1, F1, (nCh, 1), max_norm=2, bias=False),
            nn.BatchNorm2d(F1),
            
        )