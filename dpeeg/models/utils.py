# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import torch
import torch.nn as nn


class Conv2dWithNorm(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=1.0, p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, self.p, 0, self.max_norm)
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(")")
            self_repr = f", max_norm={self.max_norm}, p={self.p}"
            repr = repr[:last_bracket_index] + self_repr + ")"
        return repr


class LinearWithNorm(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm=1.0, p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, self.p, 0, self.max_norm)
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(")")
            self_repr = f", max_norm={self.max_norm}, p={self.p}"
            repr = repr[:last_bracket_index] + self_repr + ")"
        return repr


# class FocalLoss(nn.Module):
#     def __init__(
#         self,
#         alpha: float,
#         gamma: float,
#         reduction: Literal["none", "mean", "sum"] = "none",
#     ) -> None:
#         super().__init__()
#         assert reduction in [
#             "none",
#             "mean",
#             "sum",
#         ], f"reduction must be 'none', 'mean' or 'sum', but got {reduction}"
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, input: Tensor, target: Tensor):
#         ce_loss = F.cross_entropy(input, target, reduction="none")
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

#         if self.reduction == "mean":
#             return torch.mean(focal_loss)
#         elif self.reduction == "sum":
#             return torch.sum(focal_loss)
#         else:
#             return focal_loss


class LogVarLayer(nn.Module):
    r"""Temporal log-variance layer.

    The raw EEG data generally contains a large number of features along the
    time dimension and these features present the most amount of intra-class
    variance and high noise content. Therefore, to avoid overfitting of
    classification models, it is necessary to reduce the time dimensional
    features by effective extraction of the most relevant temporal information.
    Average Pooling strategies are the most commonly employed techniques for
    this purpose of reduction in feature dimensions. However, considering that
    various classes of MI differ in their spectral power (ERD/ERS), a variance
    operation, which represents the spectral power in the given time series
    becomes a more suitable option for EEG temporal characterization. So, for
    effective extraction of temporally discriminative information, we [1]_
    propose a novel Variance layer that characterizes a time series by
    computing its variance. So, in the forward pass, for any time-varying
    signal, :math:`g(t)`, the output of the Variance layer is given by:

    .. math::

        \mathbf{v}=\log\left(\frac{1}{T}\sum_{t=0}^{T-1}(g(t)-\mu)\right)

    where :math:`T` is total number of time-points, :math:`\mu` is the mean of
    :math:`g(t)` and log-transform is used to enhance the Gaussianity of the
    variance feature [2]_.

    Parameters
    ----------
    dim : int
        Time dimensional.
    clamp : tuple[float, float]
        The minimum and maximum values to which the variance is clamped.

    References
    ----------
    .. [1] R. Mane et al., “FBCNet: A Multi-view Convolutional Neural Network
        for Brain-Computer Interface,” arXiv.org. Accessed: Mar. 09, 2023.
        [Online]. vailable: https://arxiv.org/abs/2104.01233v1
    .. [2] K. Liu, M. Yang, Z. Yu, G. Wang, and W. Wu, “FBMSNet: A Filter-Bank
        Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery
        Decoding,” IEEE Trans. Biomed. Eng., vol. 70, no. 2, pp. 436-445, Feb.
        2023, doi: 10.1109/TBME.2022.3193277.
    """

    def __init__(
        self,
        dim: int,
        clamp: tuple[float, float] = (1e-6, 1e6),
    ):
        super().__init__()
        self.dim = dim
        self.clamp = clamp

    def forward(self, x):
        out = torch.clamp(x.var(dim=self.dim), self.clamp[0], self.clamp[1])
        return torch.log(out)
