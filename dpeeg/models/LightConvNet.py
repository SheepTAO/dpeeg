'''
References
----------
“A Temporal Dependency Learning CNN With Attention Mechanism for MI-EEG Decoding
| IEEE Journals & Magazine | IEEE Xplore.” Accessed: Oct. 20, 2023. [Online]. 
Available: https://ieeexplore.ieee.org/document/10196350
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out = torch.clamp(x.var(dim=self.dim), 1e-6, 1e6)
        return torch.log(out)


class LightweightConv1d(nn.Module):
    """
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """
    def __init__(self, input_size, kernel_size=1, padding=0, heads=1, 
                 weight_softmax=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.heads = heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.init_parameters()
        
    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        B, C, T = input.size()
        H = self.heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        
        input = input.view(-1, H, T)
        output = F.conv1d(
            input, weight, padding=self.padding, groups=self.heads
        )
        output = output.view(B, C, -1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        
        return output


class LightConvNet(nn.Module):
    def __init__(self, nCh=22, nTime=1000, cls=4, bands=9, embed_dim=64, 
                 win_len=250, heads=8, weight_softmax=True, bias=False):
        super().__init__()

        self.win_len = win_len
        
        self.spacial_block = nn.Sequential(
            nn.Conv2d(bands, embed_dim, (nCh, 1)),
            nn.BatchNorm2d(embed_dim),
            nn.ELU()
        )

        self.temporal_block = LogVarLayer(dim=3)

        self.conv = LightweightConv1d(
            embed_dim, (nTime//win_len), heads=heads, 
            weight_softmax=weight_softmax, bias=bias
        )

        self.classify = nn.Sequential(
            nn.Linear(embed_dim, cls),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.spacial_block(x)
        out = out.reshape([*out.shape[0:2], -1, self.win_len])
        out = self.temporal_block(out)
        out = self.conv(out)
        out = out.view(out.size(0), -1)
        out = self.classify(out)
        return out


if __name__ == '__main__':
    from torchinfo import summary
    net = LightConvNet().cuda()
    summary(net, (64, 9, 22, 1000))