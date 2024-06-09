'''
References
----------
J. Wang, L. Yao and Y. Wang, "IFNet: An Interactive Frequency Convolutional 
Neural Network for Enhancing Motor Imagery Decoding from EEG," in IEEE 
Transactions on Neural Systems and Rehabilitation Engineering, 
doi: 10.1109/TNSRE.2023.3257319.
'''


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_


class InterFre(nn.Module):
    def forward(self, x):
        out = sum(x)
        out = nn.GELU()(out)
        return out


class IFNet(nn.Module):
    def __init__(self, nCh, nTime, cls, F=64, C=63, radix=2, P=125, p=0.5) -> None:
        super().__init__()
        self.F = F
        self.mF = F * radix

        self.sConv = nn.Sequential(
            nn.Conv1d(nCh * radix, self.mF, 1, bias=False, groups=radix),
            nn.BatchNorm1d(self.mF)
        )

        self.tConv = nn.ModuleList()
        for _ in range(radix):
            self.tConv.append(nn.Sequential(
                nn.Conv1d(F, F, C, 1, padding=C//2, groups=F, bias=False),
                nn.BatchNorm1d(F)
            ))
            C //= 2

        self.interFre = InterFre()
        self.downSamp = nn.Sequential(
            nn.AvgPool1d(P), 
            nn.Dropout(p)
        )
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(int(F * (nTime // P)), cls),
            nn.LogSoftmax(dim=1)
        )

        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                constant_(m.weight, 1.0)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                constant_(m.bias, 0)

    def forward(self, x):
        out = self.sConv(x)
        out = torch.split(out, self.F, dim=1)
        out = [m(x) for m, x in zip(self.tConv, out)]
        out = self.interFre(out)
        out = self.downSamp(out)
        return self.fc(out)


class IFNetAdamW(torch.optim.AdamW):
    def __init__(self, net: nn.Module, **kwargs) -> None:
        has_decay = []
        no_decay = []

        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith('.bias'):
                no_decay.append(param)
            else:
                has_decay.append(param)
        params = [{'params': has_decay},
                  {'params': no_decay, 'weight_decay': 0}]

        super().__init__(params, **kwargs)


if __name__ == "__main__":
    from torchinfo import summary
    net = IFNet(22, 750, 4).cuda()
    summary(net, (1, 44, 750))