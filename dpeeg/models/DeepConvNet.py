import torch
import torch.nn as nn

from .utils import Conv2dWithNorm, LinearWithNorm


__all__ = ["DeepConvNet"]


class DeepConvNet(nn.Module):
    """Deep Learning With Convolutional Neural Networks for EEG Decoding and
    Visualization (Deep ConvNet).

    Deep ConvNet [1]_ had four convolution-max-pooling blocks, with a special
    first block designed to handle EEG input, followed by three standard
    convolutionmax-pooling blocks and a dense softmax classification layer.
    The first convolutional block was split into two layers in order to better
    handle the large number of input channelsone input channel per electrode
    compared to three input channels (one per color) in rgb-images. In the
    first layer, each filter performs a convolution over time, and in the
    second layer, each filter performs a spatial filtering with weights for all
    possible pairs of electrodes with filters of the preceding temporal
    convolution. Note that as there is no activation function in between the
    two layers, they could in principle be combined into one layer. Using two
    layers however implicitly regularizes the overall convolution by forcing a
    separation of the linear transformation into a combination of a temporal
    convolution and a spatial filter.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    dropout : float
        Dropout rate.

    References
    ----------
    .. [1] R. T. Schirrmeister et al., “Deep learning with convolutional neural
        networks for EEG decoding and visualization,” Human Brain Mapping,
        vol. 38, no. 11, pp.5391-5420, 2017, doi: 10.1002/hbm.23730.
    """

    def __init__(self, nCh=22, nTime=1000, cls=4, dropout=0.25) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        kernel_size = [1, 10]
        filter_layer = [25, 50, 100, 200]

        first_layer = nn.Sequential(
            Conv2dWithNorm(1, 25, kernel_size, max_norm=2),
            Conv2dWithNorm(25, 25, (nCh, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
        )
        middle_layer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Dropout(dropout),
                    Conv2dWithNorm(in_f, out_f, kernel_size),
                    nn.BatchNorm2d(out_f),
                    nn.ELU(),
                    nn.MaxPool2d((1, 3)),
                )
                for in_f, out_f in zip(filter_layer, filter_layer[1:])
            ]
        )
        self.conv_layer = nn.Sequential(first_layer, middle_layer)

        linear_in = self._forward_flatten().shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            LinearWithNorm(linear_in, cls, max_norm=0.5),
            nn.LogSoftmax(dim=1),
        )

    def _forward_flatten(self):
        x = torch.rand(1, 1, self.nCh, self.nTime)
        x = self.conv_layer(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return x

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
        x = self.conv_layer(x)
        x = self.head(x)
        return x
