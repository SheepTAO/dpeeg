'''
References
----------
Y. Song, Q. Zheng, B. Liu, and X. Gao, “EEG conformer: Convolutional transformer for EEG decoding and visualization,” 
IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710–719, 2022.
'''


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, nCh, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (nCh, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p):
        super().__init__(*[
            TransformerEncoderBlock(emb_size, num_heads, drop_p, forward_expansion, forward_drop_p)
            for _ in range(depth)
        ])


class ClassificationHead(nn.Sequential):
    def __init__(self, in_features, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, cls),
            nn.LogSoftmax(dim=1)
        )


class EEGConformer(nn.Module):
    def __init__(
        self, 
        nCh, 
        nTime, 
        cls, 
        emb_size=40, 
        depth=6, 
        num_heads=10, 
        drop_p=0.5, 
        forward_expansion=4, 
        forward_drop_p=0.5
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        self.patch_embedding = PatchEmbedding(nCh, emb_size)
        self.transformer_encoder = TransformerEncoder(
            depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p
        )
        in_freatures = self._forward_transformer().size(1)
        self.classification_head = ClassificationHead(in_freatures, cls)

    def _forward_transformer(self) -> torch.Tensor:
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return x.flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        out = self.patch_embedding(x)
        out = self.transformer_encoder(out)
        return self.classification_head(out)


if __name__ == '__main__':
    from torchinfo import summary
    net = EEGConformer(22, 1000, 4)
    summary(net, (1, 1, 22, 1000), device='cpu')