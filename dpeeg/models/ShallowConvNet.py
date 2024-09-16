import torch
import torch.nn as nn

from .utils import Conv2dWithNorm, LinearWithNorm

__all__ = ["ShallowConvNet"]


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ShallowConvNet(nn.Module):
    """Deep Learning With Convolutional Neural Networks for EEG Decoding and
    Visualization (ShallowConvNet).

    Shallow ConvNet [1]_, inspired by the FBCSP pipeline, is specifically
    tailored to decode band power features. The transformations performed by
    the shallow ConvNet are similar to the transformations of FBCSP.
    Concretely, the first two layers of the shallow ConvNet perform a temporal
    convolution and a spatial filter, as in the deep ConvNet. These steps are
    analogous to the bandpass and CSP spatial filter steps in FBCSP. In
    contrast to the deep ConvNet, the temporal convolution of the shallow
    ConvNet had a larger kernel size, allowing a larger range of
    transformations in this layer (smaller kernel sizes for the shallow ConvNet
    led to lower accuracies in preliminary experiments on the training set).
    After the temporal convolution and the spatial filter of the shallow
    ConvNet, a squaring nonlinearity, a mean pooling layer and a logarithmic
    activation function followed; together these steps are analogous to the
    trial log-variance computation in FBCSP. In contrast to FBCSP, the shallow
    ConvNet embeds all the computational steps in a single network, and thus
    all steps can be optimized jointly. Also, due to having several pooling
    regions within one trial, the shallow ConvNet can learn a temporal
    structure of the band power changes within the trial.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    F : int
        The number of convolution channels.
    C : int
        Temporal convolution kernel size.
    P : int
        Pooling kernel size.
    S : int
        Pooling layer stride size.
    dropout : float
        Dropout rate.

    References
    ----------
    .. [1] R. T. Schirrmeister et al., “Deep learning with convolutional neural
        networks for EEG decoding and visualization,” Human Brain Mapping,
        vol. 38, no. 11, pp.5391-5420, 2017, doi: 10.1002/hbm.23730.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        cls: int,
        F: int = 40,
        C: int = 14,
        P: int = 35,
        S: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime

        self.conv = nn.Sequential(
            Conv2dWithNorm(1, F, (1, C), max_norm=2, bias=False),
            Conv2dWithNorm(F, F, (nCh, 1), max_norm=2, bias=False, groups=F),
            nn.BatchNorm2d(F),
            Lambda(torch.square),
            nn.AvgPool2d((1, P), stride=(1, S)),
            Lambda(torch.log),
        )

        linear_in = self.forward_flatten().shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            LinearWithNorm(linear_in, cls, max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        x = self.conv(x)
        x = torch.flatten(x, 1, -1)
        return x

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
        x = self.conv(x)
        x = self.head(x)
        return x
