'''
    EEGNet-8,2
    https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
'''


import torch
from torch import Tensor, nn
from torchinfo import summary


class Conv2dWithNorm(nn.Conv2d):
    
    def __init__(self, *args, maxNorm, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.maxNorm = maxNorm
        
    def forward(self, input: Tensor) -> Tensor:
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
    
    def __init__(self, nCh, nTime, kernel=125, F1=8, D=2, F2=16,
                 p=0.25, cls=4, bias=False) -> None:
        super().__init__()
        
        self.filter = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel), padding='same', bias=bias),
            nn.BatchNorm2d(F1)
        )
        
        self.depthwiseConv = nn.Sequential(
            Conv2dWithNorm(F1, D*F1, (nCh, 1), groups=F1, bias=bias, maxNorm=1),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(p)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, 16), padding='same', groups=D*F1, bias=bias),
            nn.Conv2d(D*F1, F2, 1, bias=bias),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout2d(p)
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            LinearWithNorm(self.get_size(nCh, nTime), cls, bias=bias, maxNorm=0.25),
            nn.LogSoftmax(dim=1)
        )
        
    def get_size(self, nCh, nTime):
        
        x = torch.randn(1, 1, nCh, nTime)
        out = self.filter(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = self.flatten(out)
        
        return out.size(1)

    def forward(self, x):
        
        out = self.filter(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = self.flatten(out)
        
        return self.fc(out)
    

if __name__ == '__main__':
    net = EEGNet(22, 1125).cuda()
    summary(net, (1, 1, 22, 1125))