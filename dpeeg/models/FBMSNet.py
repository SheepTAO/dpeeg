import torch
from torch import nn
from torch.autograd.function import Function
from torch.optim import Optimizer, SGD, Adam

from .utils import Conv2dWithNorm, LinearWithNorm, LogVarLayer
from ..tools.docs import fill_doc


def split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="same"):
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = split_channels(in_channels, num_groups)
        out_splits = split_channels(out_channels, num_groups)

        self.conv = nn.ModuleList()
        for k, in_ch, out_ch in zip(kernel_size, in_splits, out_splits):
            self.conv.append(nn.Conv2d(in_ch, out_ch, k, padding=padding))
        self.splits = in_splits

    def forward(self, X):
        X_splits = torch.split(X, self.splits, 1)
        out = [conv(X_splits[i]) for i, conv in enumerate(self.conv)]
        out = torch.cat(out, 1)
        return out


@fill_doc
class FBMSNet(nn.Module):
    """FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for
    EEG-Based Motor Imagery Decoding

    FBMSNet [1]_ consists of four blocks, (1) a temporal convolution block,
    (2) a spatial convolution block, (3) a temporal logvariance block, and
    (4) a fully connected layer for classification. The first block is designed
    to learn the multiscale temporal information from the multiview EEG
    representations, and the second block aims to learn the spatial information
    from each temporal feature map. Subsequently, the third block computes the
    temporal variance of each time series. Finally, all representations are
    flattened and fed to the fully connected layer with softmax as the
    activation function.

    Parameters
    ----------
    %(nCh)s
    %(nTime)s
    %(nCls)s
    branch_size : list of int
        Multi-scale temporal convolution kernel sizes. The default is 0.0625,
        0.125, 0.25 and 0.5 ratios for a sampling rate of 250 Hz.
    bands : int
        The filter dimension of the input multi-view data.
    num_feat : int
        Output feature map dimensions of multi-scale temporal convolutional
        layers. The default is 4 branches with 9 in each branch.
    dilatability : int
        Spatial convolution dilation degree.
    stride : int
        Number of time windows. Must evenly divide `nTime`.

    References
    ----------
    .. [1] K. Liu, M. Yang, Z. Yu, G. Wang, and W. Wu, “FBMSNet: A Filter-Bank
        Multi-Scale Convolutional Neural Network for EEG-Based Motor Imagery
        Decoding,” IEEE Trans. Biomed. Eng., vol. 70, no. 2, pp. 436-445, Feb.
        2023, doi: 10.1109/TBME.2022.3193277.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        nCls: int,
        branch_size: list[int] = [15, 31, 63, 125],
        bands: int = 9,
        num_feat: int = 36,
        dilatability: int = 8,
        stride: int = 4,
    ) -> None:
        super().__init__()
        assert not (nTime % stride), "nTime must be divisible by stride."
        self.nCh = nCh
        self.nTime = nTime
        self.bands = bands
        self.stride = stride

        kernel_size = list(zip([1] * len(branch_size), branch_size))
        self.tempConv = nn.Sequential(
            MixedConv2d(
                in_channels=bands, out_channels=num_feat, kernel_size=kernel_size
            ),
            nn.BatchNorm2d(num_feat),
        )

        self.spatConv = nn.Sequential(
            Conv2dWithNorm(
                in_channels=num_feat,
                out_channels=num_feat * dilatability,
                kernel_size=(nCh, 1),
                groups=num_feat,
                max_norm=2,
            ),
            nn.BatchNorm2d(num_feat * dilatability),
            swish(),
        )

        self.tempLogVar = LogVarLayer(dim=3)

        feat_size = self.get_feat_size()
        self.cls = nn.Sequential(
            nn.Flatten(),
            LinearWithNorm(in_features=feat_size, out_features=nCls, max_norm=0.5),
            nn.LogSoftmax(dim=-1),
        )

        self.center = nn.Parameter(torch.randn(nCls, feat_size))

    @torch.no_grad()
    def get_feat_size(self):
        x = torch.randn((1, self.bands, self.nCh, self.nTime))
        out = self.tempConv(x)
        out = self.spatConv(out)
        out = out.view(*out.size()[:2], self.stride, out.size()[3] // self.stride)
        out = self.tempLogVar(out)
        return torch.flatten(out, start_dim=1).size(1)

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
            Predicted class probability, shape `(batch_size, nCls)`.
        feature : Tensor
            Feature vector before classification layer, used for center loss.
        center : Tensor
            Feature center.
        """
        out = self.tempConv(x)
        out = self.spatConv(out)
        out = out.view(*out.size()[:2], self.stride, out.size()[3] // self.stride)
        out = self.tempLogVar(out)
        feature = torch.flatten(out, start_dim=1)
        return self.cls(out), feature, self.center


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        batch_size = feature.new_empty(1).fill_(feature.size(0))
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(
            0, label.unsqueeze(1).expand(feature.size()).long(), diff
        )
        grad_centers = grad_centers / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CECLoss(nn.Module):
    r"""FBMSNet: A Filter-Bank Multi-Scale Convolutional Neural Network for
    EEG-Based Motor Imagery Decoding (Cross-Entropy Loss + Center Loss)

    The parameters of FBMSNet are learned under the supervision of the
    cross-entropy loss and center loss:

    .. math::

        \mathcal{L}_{total} = \mathcal{L}_{CE}+\lambda\mathcal{L}_C

    where center loss updates the feature centers based on the mini-batch in
    each iteration.
    """

    def __init__(self, lamd: float) -> None:
        super().__init__()
        self.lamd = lamd
        self.celoss = nn.NLLLoss()
        self.closs = CenterlossFunc.apply

    def forward(self, out, target):
        """Forward pass function that processes cross-entropy loss and center loss.

        Parameters
        ----------
        out : tuple of Tensor
            Tuple of cls_prob (predicted class probability), feat (feature
            vector before classification layer) and center (feature center).
        target : Tensor
            True label.
        """
        cls_prob, feat, center = out
        celoss = self.celoss(cls_prob, target)
        closs = self.closs(feat, target, center)
        return celoss + self.lamd * closs  # type: ignore


class CECOptim(Optimizer):
    """Optimzer for CECLoss.

    The cross-entropy loss and the center loss are optimized by the Adam and
    SGD optimizers, respectively.

    Parameters
    ----------
    params : list of Tensor
        Parameters of FBMSNet.
    lr : float
        Learning rate of Adam optimizer.
    alpha : float
        Learning rate of SGD optimizer for center.
    """

    def __init__(self, params, lr, alpha=0.01) -> None:
        params = list(params)
        ceparams = params[1:]  # parameters of FBMSNet without center
        cparams = params[0]  # parameters of center
        self.ceoptim = Adam(ceparams, lr)
        self.coptim = SGD(cparams, alpha)
        super().__init__(params, {})

    def zero_grad(self) -> None:
        self.ceoptim.zero_grad()
        self.coptim.zero_grad()

    def step(self):
        self.ceoptim.step()
        self.coptim.step()

    def state_dict(self):
        return {
            "ceoptim": self.ceoptim.state_dict(),
            "coptim": self.coptim.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.ceoptim.load_state_dict(state_dict["ceoptim"])
        self.coptim.load_state_dict(state_dict["coptim"])
