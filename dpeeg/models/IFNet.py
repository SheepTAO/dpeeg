import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.nn.init import trunc_normal_, constant_

__all__ = ["IFNet", "IFNetAdamW"]


class InterFre(nn.Module):
    def forward(self, x):
        out = sum(x)
        out = nn.GELU()(out)
        return out


class IFNet(nn.Module):
    """IFNet: An Interactive Frequency Convolutional Neural Network for
    Enhancing Motor Imagery Decoding From EEG (IFNet).

    Inspired by the concept of cross-frequency coupling and its correlation
    with different behavioral tasks, IFNet [1]_ explores cross-frequency
    interactions for enhancing representation of MI characteristics. IFNet
    first extracts spectro-spatial features in low and high-frequency bands,
    respectively. Then the interplay between the two bands is learned using an
    element-wise addition operation followed by temporal average pooling.
    Combined with repeated trial augmentation as a regularizer, IFNet yields
    spectro-spatiotemporally robust features for the final MI classification.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    F : int
        Number of spectro-spatial filters.
    C : int
        Spectro-spatial filter kernel size.
    radix : int
        Number of cross-frequency domains.
    P : int
        Pooling kernel size.
    dropout : float
        Dropout rate.

    References
    ----------
    .. [1] J. Wang, L. Yao and Y. Wang, "IFNet: An Interactive Frequency
        Convolutional Neural Network for Enhancing Motor Imagery Decoding from
        EEG," in IEEE Transactions on Neural Systems and Rehabilitation
        Engineering, doi: 10.1109/TNSRE.2023.3257319.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        cls: int,
        F: int = 64,
        C: int = 63,
        radix: int = 2,
        P: int = 125,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.F = F
        self.mF = F * radix

        self.sConv = nn.Sequential(
            nn.Conv1d(nCh * radix, self.mF, 1, bias=False, groups=radix),
            nn.BatchNorm1d(self.mF),
        )

        self.tConv = nn.ModuleList()
        for _ in range(radix):
            self.tConv.append(
                nn.Sequential(
                    nn.Conv1d(F, F, C, 1, padding=C // 2, groups=F, bias=False),
                    nn.BatchNorm1d(F),
                )
            )
            C //= 2

        self.interFre = InterFre()
        self.downSamp = nn.Sequential(nn.AvgPool1d(P), nn.Dropout(dropout))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(int(F * (nTime // P)), cls), nn.LogSoftmax(dim=1)
        )

        self.apply(self.initParms)

    def initParms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                constant_(m.weight, 1.0)
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass function that processes the input EEG data and produces
        the decoded results.

        Parameters
        ----------
        x : Tensor
            Input EEG data, shape `(batch_size, nCh * radix, nTime)`.

        Returns
        -------
        cls_prob : Tensor
            Predicted class probability, shape `(batch_size, cls)`.
        """
        out = self.sConv(x)
        out = torch.split(out, self.F, dim=1)
        out = [m(x) for m, x in zip(self.tConv, out)]
        out = self.interFre(out)
        out = self.downSamp(out)
        return self.fc(out)


class IFNetAdamW(AdamW):
    """Customized AdamW Optimizer for IFNet.

    IFNetAdamW optimizer allows bias and weights based on certain parameters to
    not decay.

    Parameters
    ----------
    net : IFNet
        IFNet model instance.
    """

    def __init__(self, net: nn.Module, **kwargs) -> None:
        has_decay = []
        no_decay = []

        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                has_decay.append(param)
        params = [{"params": has_decay}, {"params": no_decay, "weight_decay": 0}]

        super().__init__(params, **kwargs)
