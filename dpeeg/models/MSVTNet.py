import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

__all__ = ["MSVTNet", "JointCrossEntoryLoss"]


class TSConv(nn.Sequential):
    def __init__(self, nCh, F, C1, C2, D, P1, P2, Pc) -> None:
        super().__init__(
            nn.Conv2d(1, F, (1, C1), padding="same", bias=False),
            nn.BatchNorm2d(F),
            nn.Conv2d(F, F * D, (nCh, 1), groups=F, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(Pc),
            nn.Conv2d(F * D, F * D, (1, C2), padding="same", groups=F * D, bias=False),
            nn.BatchNorm2d(F * D),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(Pc),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x += self.pe
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        nhead,
        ff_ratio,
        Pt=0.5,
        num_layers=4,
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = PositionalEncoding(seq_len + 1, d_model)

        dim_ff = d_model * ff_ratio
        self.dropout = nn.Dropout(Pt)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_ff, Pt, batch_first=True, norm_first=True
            ),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class ClsHead(nn.Sequential):
    def __init__(self, linear_in, cls):
        super().__init__(nn.Flatten(), nn.Linear(linear_in, cls), nn.LogSoftmax(dim=1))


class MSVTNet(nn.Module):
    """MSVTNet: Multi-Scale Vision Transformer Neural Network for EEG-Based
    Motor Imagery Decoding (MSVTNet).

    MSVTNet [1]_ effectively integrates the strengths of convolutional neural
    networks (CNNs) in extracting local features with the global feature
    extraction capabilities of Transformers. Specifically, to optimize
    classification features, a multi-branch CNN with different scales is
    designed to capture local spatiotemporal features, along with a Transformer
    to jointly model global and local spatiotemporal correlations features.
    Additionally, auxiliary branch loss (ABL) is leveraged for intermediate
    supervision, ensuring effective integration of CNNs and Transformers.

    Parameters
    ----------
    nCh : int
        Number of electrode channels.
    nTime : int
        Number of data sampling points.
    cls : int
        Number of categories.
    F : list of int
        Number of temporal filters per branch.
    C1 : list of int
        The convolution kernel size of each branch temporal filter.
    C2 : int
        Depthwise convolution kernel size.
    D : int
        Depth of depthwise convolution.
    P1 : float
        The first pooling kernel size.
    P2 : float
        The second pooling kernel size.
    Pc : float
        Dropout rate of multi-branch convolutional module.
    nhead : int
        Number of multi-head attention.
    ff_ratio : int
        The expansion factor of the fully connected feed-forward layer.
    Pt : float
        Dropout rate of transformer encoder.
    layers : int
        Number of transformer encoder layers.
    b_preds : bool
        If ``True``, return the prediction for each branch.

    References
    ----------
    .. [1] K. Liu et al., "MSVTNet: Multi-Scale Vision Transformer Neural
        Network for EEG-Based Motor Imagery Decoding," in IEEE Journal of
        Biomedical and Health Informatics, doi: 10.1109/JBHI.2024.3450753.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        cls: int,
        F: list[int] = [9, 9, 9, 9],
        C1: list[int] = [15, 31, 63, 125],
        C2: int = 15,
        D: int = 2,
        P1: int = 8,
        P2: int = 7,
        Pc: float = 0.3,
        nhead: int = 8,
        ff_ratio: int = 1,
        Pt: float = 0.5,
        layers: int = 2,
        b_preds: bool = True,
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        self.b_preds = b_preds
        assert len(F) == len(C1), "The length of F and C1 should be equal."

        self.mstsconv = nn.ModuleList(
            [
                nn.Sequential(
                    TSConv(nCh, F[b], C1[b], C2, D, P1, P2, Pc),
                    Rearrange("b d 1 t -> b t d"),
                )
                for b in range(len(F))
            ]
        )
        branch_linear_in = self._forward_flatten(cat=False)
        self.branch_head = nn.ModuleList(
            [ClsHead(branch_linear_in[b].shape[1], cls) for b in range(len(F))]
        )

        seq_len, d_model = self._forward_mstsconv().shape[1:3]  # type: ignore
        self.transformer = Transformer(seq_len, d_model, nhead, ff_ratio, Pt, layers)

        linear_in = self._forward_flatten().shape[1]  # type: ignore
        self.last_head = ClsHead(linear_in, cls)

    def _forward_mstsconv(self, cat=True):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = [tsconv(x) for tsconv in self.mstsconv]
        if cat:
            x = torch.cat(x, dim=2)
        return x

    def _forward_flatten(self, cat=True):
        x = self._forward_mstsconv(cat)
        if cat:
            x = self.transformer(x)
            x = x.flatten(start_dim=1, end_dim=-1)
        else:
            x = [_.flatten(start_dim=1, end_dim=-1) for _ in x]
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
        branch_cls_prob : list of Tensor
            If ``b_preds=True``, return the class prediction probability for
            each branch.
        """
        x = [tsconv(x) for tsconv in self.mstsconv]
        bx = [branch(x[idx]) for idx, branch in enumerate(self.branch_head)]
        x = torch.cat(x, dim=2)
        x = self.transformer(x)
        x = self.last_head(x)
        if self.b_preds:
            return x, bx
        else:
            return x


class JointCrossEntoryLoss(nn.Module):
    r"""Auxiliary branch loss.

    The parameters of MSVTNet are learned under the supervision of the
    auxiliary branch loss and model prediction loss:

    .. math::

       \mathcal{L}=\lambda\mathcal{L}_c+(1-\lambda)\sum_{b=1}^{B}\mathcal{L}_b

       \mathcal{L}_{c/b}=\mathrm{Cross Entropy Loss}(\hat{y})

    where :math:`\lambda\in(0, 1]` is the ratio factor for intermediate
    supervision of the model.

    Parameters
    ----------
    lamd : float
        Ratio factor of ABL.
    """

    def __init__(self, lamd: float = 0.6) -> None:
        super().__init__()
        self.lamd = lamd

    def forward(self, out, label):
        """Forward pass function that processes the model and branch prediction
        probabilities.

        Parameters
        ----------
        out : tuple of Tensor
            Models and branch prediction probabilities.
        label : Tensor
            True label.

        Returns
        -------
        loss : Tensor
            Loss with gradient.
        """
        end_out = out[0]
        branch_out = out[1]
        end_loss = F.nll_loss(end_out, label)
        branch_loss = [F.nll_loss(out, label).unsqueeze(0) for out in branch_out]
        branch_loss = torch.cat(branch_loss)
        loss = self.lamd * end_loss + (1 - self.lamd) * torch.sum(branch_loss)
        return loss
