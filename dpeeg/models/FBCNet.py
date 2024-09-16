import torch
from torch import nn

from .utils import Conv2dWithNorm, LinearWithNorm


__all__ = ["FBCNet"]


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6)
        return torch.log(x)


class FBCNet(nn.Module):
    """FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer
    Interface (FBCNet).

    FBCNet [1]_ employs a multi-view data representation followed by spatial
    filtering to extract spectro-spatially discriminative features. In FBCNet,
    a novel Variance layer is proposed to effectively aggregate the EEG time-
    domain information.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    bands : int
        The filter dimension of the input multi-view data.
    m : int
        Number of sptatial filters.
    stride : int
        Number of time windows. Must evenly divide `nTime`.

    References
    ----------
    .. [1] R. Mane et al., “FBCNet: A Multi-view Convolutional Neural Network
        for Brain-Computer Interface,” arXiv.org. Accessed: Mar. 09, 2023.
        [Online]. vailable: https://arxiv.org/abs/2104.01233v1
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        cls: int,
        bands: int = 9,
        m: int = 32,
        stride: int = 4,
    ) -> None:
        super().__init__()
        assert not (nTime % stride), "nTime must be divisible by stride."
        self.stride = stride

        self.scb = nn.Sequential(
            Conv2dWithNorm(
                bands,
                m * bands,
                (nCh, 1),
                groups=bands,
                do_weight_norm=True,
                max_norm=2,
                padding=0,
            ),
            nn.BatchNorm2d(m * bands),
            swish(),
        )

        self.temporal_layer = LogVarLayer(dim=3)

        self.head = nn.Sequential(
            nn.Flatten(),
            LinearWithNorm(m * bands * stride, cls, do_weight_norm=True, max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        """Forward pass function that processes the input EEG data and produces
        the decoded results.

        Parameters
        ----------
        x : Tensor
            Input EEG data, shape `(batch_size, bands, nCh, nTime)`.

        Returns
        -------
        cls_prob : Tensor
            Predicted class probability, shape `(batch_size, cls)`.
        """
        x = self.scb(x)
        x = x.reshape([*x.shape[:2], self.stride, x.shape[3] // self.stride])
        x = self.temporal_layer(x)
        x = self.head(x)
        return x
