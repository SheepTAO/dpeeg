import torch
import torch.nn as nn

from .utils import Conv2dWithNorm
from ..tools.docs import fill_doc


__all__ = ["EEGInception"]


@fill_doc
class EEGInception(nn.Module):
    """EEG-Inception: A Novel Deep Convolutional Neural Network for Assistive
    ERP-Based Brain-Computer Interfaces

    EEG-Inception [1]_ includes several concepts adapted from the image
    classification domain, including Inception modules to capture dependencies
    between features at different scales and depthwise convolutions.

    .. warning::

       This is an unofficial PyTorch implementation, which may contain
       discrepancies. Please refer to the official implementation:
       https://github.com/esantamariavazquez/EEG-Inception

    Parameters
    ----------
    %(nCh)s
    %(nTime)s
    %(nCls)s
    filters_per_branch : int
        Number of filters in each Inception branch.
    scales_time : list of int
        Temporal scale (sampling points) of the convolutions on each Inception
        module. This parameter determines the kernel sizes of the filters, and
        the original design was to capture features from 500 ms, 250 ms and 100
        ms time windows.
    dropout_rate : float
        Dropout rate.

    References
    ----------
    .. [1] Santamaria-Vazquez, E., Martinez-Cagigal, V., Vaquerizo-Villar, F.,
        & Hornero, R. (2020). EEG-inception: a novel deep convolutional neural
        network for assistive ERP-based brain-computer interfaces.
        IEEE Transactions on Neural Systems and Rehabilitation Engineering,
        28(12), 2773-2782.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        nCls: int,
        scales_time: list[int] = [64, 32, 16],
        filters_per_branch: int = 8,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime

        self.inception_1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, filters_per_branch, (1, scale), padding="same"),
                    nn.BatchNorm2d(filters_per_branch),
                    nn.ELU(),
                    nn.Dropout(dropout_rate),
                    Conv2dWithNorm(
                        filters_per_branch,
                        filters_per_branch * 2,
                        (nCh, 1),
                        groups=filters_per_branch,
                        bias=False,
                        max_norm=1,
                    ),
                    nn.BatchNorm2d(filters_per_branch * 2),
                    nn.ELU(),
                    nn.Dropout(dropout_rate),
                )
                for scale in scales_time
            ]
        )
        nn.init.kaiming_normal_(
            self.inception_1[0][0].weight, mode="fan_out", nonlinearity="relu"  # type: ignore
        )
        self.avgpool_1 = nn.AvgPool2d((1, 4))

        self.inception_2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        filters_per_branch * 2 * len(scales_time),
                        filters_per_branch,
                        (1, scale // 4),
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm2d(filters_per_branch),
                    nn.ELU(),
                    nn.Dropout(dropout_rate),
                )
                for scale in scales_time
            ]
        )
        nn.init.kaiming_normal_(
            self.inception_2[0][0].weight, mode="fan_out", nonlinearity="relu"  # type: ignore
        )
        self.avgpool_2 = nn.AvgPool2d((1, 2))

        self.conv = nn.Sequential(
            nn.Conv2d(
                filters_per_branch * len(scales_time),
                filters_per_branch * len(scales_time) // 2,
                (1, 8),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_time) // 2),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                filters_per_branch * len(scales_time) // 2,
                filters_per_branch * len(scales_time) // 4,
                (1, 4),
                bias=False,
                padding="same",
            ),
            nn.BatchNorm2d(filters_per_branch * len(scales_time) // 4),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(dropout_rate),
        )
        nn.init.kaiming_normal_(
            self.conv[0].weight, mode="fan_out", nonlinearity="relu"  # type: ignore
        )
        nn.init.kaiming_normal_(
            self.conv[5].weight, mode="fan_out", nonlinearity="relu"  # type: ignore
        )

        _, f, _, t = self._forward_cls()
        self.cls = nn.Sequential(
            nn.Flatten(), nn.Linear(f * t, nCls), nn.LogSoftmax(dim=-1)
        )

    @torch.no_grad()
    def _forward_cls(self):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        out = torch.cat([inception(x) for inception in self.inception_1], dim=1)
        out = self.avgpool_1(out)
        out = torch.cat([inception(out) for inception in self.inception_2], dim=1)
        out = self.avgpool_2(out)
        return self.conv(out).size()

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
            Predicted class probability, shape `(batch_size, nCls)`.
        """
        out = torch.cat([inception(x) for inception in self.inception_1], dim=1)
        out = self.avgpool_1(out)
        out = torch.cat([inception(out) for inception in self.inception_2], dim=1)
        out = self.avgpool_2(out)
        out = self.conv(out)
        return self.cls(out)
