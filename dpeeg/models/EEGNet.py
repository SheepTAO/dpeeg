import torch
import torch.nn as nn

from .utils import Conv2dWithNorm, LinearWithNorm


__all__ = ["EEGNet"]


class EEGNet(nn.Module):
    """EEGNet: A Compact Convolutional Neural Network for EEG-based
    Brain-Computer Interfaces (EEGNet).

    EEGNet [1]_ is a compact convolutional neural network for EEG-based BCIs.
    EEGNet starts with a temporal convolution to learn frequency filters, then
    uses a depthwise convolution, connected to each feature map individually,
    to learn frequency-specific spatial filters. The separable convolution is a
    combination of a depthwise convolution, which learns a temporal summary for
    each feature map individually, followed by a pointwise convolution, which
    learns how to optimally mix the feature maps together.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    F1 : int
        Number of temporal filters.
    C1 : int
        Temporal convolution kernel size.
    D : int
        Depth of depthwise convolution.
    F2 : int
        Number of separable convolutions.
    C2 : int
        Separable convolution kernel size.
    P1 : int
        The first pooling kernel size.
    P2 : int
        The second pooling kernel size.
    dropout : float
        Dropout rate.

    References
    ----------
    .. [1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon,
        C. P. Hung, and B. J. Lance, “EEGNet: a compact convolutional neural
        network for EEG-based brain–computer interfaces,” J. Neural Eng.,
        vol. 15, no. 5, p. 056013, Jul. 2018, doi: 10.1088/1741-2552/aace8c.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        cls=4,
        F1: int = 8,
        C1: int = 63,
        D: int = 2,
        F2: int = 16,
        C2: int = 15,
        P1: int = 8,
        P2: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.filter = nn.Sequential(
            nn.Conv2d(1, F1, (1, C1), padding=(0, C1 // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.depthwise_conv = nn.Sequential(
            Conv2dWithNorm(F1, D * F1, (nCh, 1), groups=F1, bias=False, max_norm=1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(dropout),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                D * F1, D * F1, (1, C2), padding=(0, C2 // 2), groups=D * F1, bias=False
            ),
            nn.Conv2d(D * F1, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(dropout),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            # Experimental results show that using linearwithnorm will lead to
            # performance degradation.
            # LinearWithNorm(self.get_size(nCh, nTime), cls, bias=True, max_norm=0.25)
            nn.Linear(self.get_size(nCh, nTime), cls, bias=True),
            nn.LogSoftmax(dim=1),
        )

    def get_size(self, nCh, nTime):
        x = torch.randn(1, 1, nCh, nTime)
        out = self.filter(x)
        out = self.depthwise_conv(out)
        out = self.separable_conv(out)
        return self.flatten(out).size(1)

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
        out = self.filter(x)
        out = self.depthwise_conv(out)
        out = self.separable_conv(out)
        out = self.flatten(out)
        return self.fc(out)
