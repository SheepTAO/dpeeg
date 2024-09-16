import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


__all__ = ["EEGConformer"]


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
            Rearrange("b e (h) (w) -> b (h w) e"),
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
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
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
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(
        self, depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p
    ):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size, num_heads, drop_p, forward_expansion, forward_drop_p
                )
                for _ in range(depth)
            ]
        )


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
            nn.LogSoftmax(dim=1),
        )


class EEGConformer(nn.Module):
    """EEG Conformer: Convolutional Transformer for EEG Decoding and
    Visualization (EEG Conformer).

    EEG Conformer [1]_ is proposed to encapsulate local and global features in
    a unified EEG classification framework. The architecture comprises three
    components: a convolution module, a self-attention module, and a fully-
    connected classifier. In the convolution module, taking the raw two-
    dimensional EEG trials as the input, temporal and spatial convolutional
    layers are applied along the time dimension and electrode channel
    dimensions, respectively. Then, an average pooling layer is utilized to
    suppress noise interference while improving generalization. Secondly, the
    spatial-temporal representation obtained by the convolution module is fed
    into the selfattention module. The self-attention module further extracts
    the long-term temporal features by measuring the global correlations
    between different time positions in the feature maps. Finally, a compact
    classifier consisting of several fullyconnected layers is adopted to output
    the decoding results.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    emb_size : int
        Embedding layer size.
    depth : int
        Depth of transformer encoder.
    num_heads : int
        Number of multi-head attention.
    drop_p : float
        Dropout rate of transformer encoder.
    forward_expansion : int
        The expansion factor of the fully connected feed-forward layer.
    forward_drop_p : float
        Dropout rate of fully connected feed-forward layer.

    References
    ----------
    .. [1] Y. Song, Q. Zheng, B. Liu, and X. Gao, “EEG conformer: Convolutional
        transformer for EEG decoding and visualization,” IEEE Transactions on
        Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710–719,
        2022.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        cls: int,
        emb_size: int = 40,
        depth: int = 6,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.5,
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
        """Forward pass function that processes the input EEG data and produces
        the decoded results.

        Parameters
        ----------
        x : Tensor
            Input EEG data, shape `(batch_size, 1, nCh, nTime)`.

        Returns
        -------
        cls_prob : Tensor
            Predicted class probability, shape `(batch_size, cls)`.
        """
        out = self.patch_embedding(x)
        out = self.transformer_encoder(out)
        return self.classification_head(out)
