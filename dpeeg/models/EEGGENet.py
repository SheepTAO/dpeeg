import torch
import torch.nn as nn
from torch.nn import init
from einops.layers.torch import Rearrange

from .utils import Conv2dWithNorm
from ..tools.docs import fill_doc


def normalize_adj(adj, mode="sym"):
    assert len(adj.shape) in [2, 3]
    if mode == "sym":
        inv_sqrt_degree = 1.0 / (
            torch.sqrt(adj.abs().sum(dim=-1, keepdim=False)) + 1e-10
        )
        if len(adj.shape) == 3:
            return inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1.0 / (adj.abs().sum(dim=-1, keepdim=False) + 1e-10)
        if len(adj.shape) == 3:
            return inv_degree[:, :, None] * adj
        return inv_degree[:, None] * adj
    else:
        exit("wrong norm mode")


class GraphEmbedding(nn.Module):
    def __init__(self, n_nodes, adj, k=1, adj_learn=True) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k

        self.xs, self.ys = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        node_value = adj[self.xs, self.ys]
        self.edge_weight = nn.Parameter(
            node_value.clone().detach(), requires_grad=adj_learn
        )

    def forward(self, x):
        edge_weight = torch.zeros([self.n_nodes, self.n_nodes], device=x.device)
        edge_weight[self.xs.to(x.device), self.ys.to(x.device)] = self.edge_weight.to(
            x.device
        )
        edge_weight = (
            edge_weight
            + edge_weight.T
            + torch.eye(self.n_nodes, dtype=edge_weight.dtype, device=x.device)
        )
        edge_weight = normalize_adj(edge_weight, mode="sym")
        x_out = [x]
        for k in range(self.k):
            x = torch.matmul(edge_weight.unsqueeze(0), x)
            x_out.append(x)
        x_out = torch.stack(x_out)
        return x_out


@fill_doc
class EEGGENet(nn.Module):
    """EEG_GENet: A feature-level graph embedding method for motor imagery
    classification based on EEG signals

    This paper proposes a feature-level graph embedding method and combines the
    method with the existing EEGNet; the new network is called EEG_GENet [1]_.
    First, based on the spatial position of the EEG electrodes or Pearson's
    correlation matrix, an adjacency matrix is built as a graph filter. Then,
    time-domain features are obtained by convoluting each electrode's raw motor
    imagery EEG signals. Furthermore, the normalized graph filter and the time-
    domain features perform matrix multiplication to embed the topology
    information into the features. Finally, the features continue to propagate
    forward through the remaining network layers.

    Parameters
    ----------
    %(nCh)s
    %(nTime)s
    %(nCls)s
    adj :
    kernel_length : int
        The size of the temporal convolution kernel.
    F1 : int
        The number of temporal convolution filters.
    k : int
        The order of graph embedding.
    D : int
        The depth of the depthwise spatial convolution.
    F2 : int
        The number of separable convolution kernel.
    dropout_rate : float
        Dropout rate.
    adj_learn : bool
        If True, the adjacency matrix is trainable.

    References
    ----------
    .. [1] H. Wang, H. Yu, and H. Wang, “EEG_GENet: A feature-level graph
        embedding method for motor imagery classification based on EEG
        signals,” Biocybern. Biomed. Eng., vol. 42, no. 3, pp.
        1023-1040, Jul. 2022, doi: 10.1016/j.bbe.2022.08.003.
    """

    def __init__(
        self,
        nCh: int,
        nTime: int,
        nCls: int,
        adj,
        kernel_length: int = 64,
        F1: int = 8,
        k: int = 2,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.25,
        adj_learn: bool = False,
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        self.nCls = nCls
        self.k = k

        self.temporal_conv = nn.Sequential(
            Conv2dWithNorm(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
                do_weight_norm=False,
            ),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
        )

        self.graph_embedding = nn.Sequential(
            Rearrange("b f c t -> (b f) c t"),
            GraphEmbedding(nCh, adj=adj, k=k, adj_learn=adj_learn),
        )

        self.spatial_conv = nn.Sequential(
            Conv2dWithNorm(
                in_channels=F1 * (k + 1),
                out_channels=F1 * D * (k + 1),
                kernel_size=(nCh, 1),
                groups=F1 * (k + 1),
                bias=False,
                max_norm=1,
                do_weight_norm=True,
            ),
            nn.BatchNorm2d(F1 * D * (k + 1), momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
        )

        self.separable_conv = nn.Sequential(
            nn.Dropout(dropout_rate),
            Conv2dWithNorm(
                in_channels=F1 * D * (k + 1),
                out_channels=F1 * D * (k + 1),
                kernel_size=(1, 16),
                groups=F1 * D * (k + 1),
                padding=(0, 8),
                bias=False,
                do_weight_norm=False,
            ),
            Conv2dWithNorm(
                in_channels=F1 * D * (k + 1),
                out_channels=F2,
                kernel_size=1,
                bias=False,
                do_weight_norm=False,
            ),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
        )

        t = self.forward_cls().size(3)
        self.cls = nn.Sequential(
            nn.Dropout(dropout_rate),
            Conv2dWithNorm(
                in_channels=F2,
                out_channels=nCls,
                kernel_size=(1, t),
                bias=True,
                max_norm=0.25,
                do_weight_norm=True,
            ),
            nn.Flatten(),
            nn.LogSoftmax(dim=-1),
        )

        self.apply(self.glorot_weight_zero_bias)

    @torch.no_grad()
    def forward_cls(self):
        x = torch.randn(1, 1, self.nCh, self.nTime)
        x = self.temporal_conv(x)
        x = self.graph_embedding(x)
        x = x.view((self.k + 1), 1, -1, x.size(-2), x.size(-1))
        x = x.permute(1, 2, 0, 3, 4).contiguous().view(1, -1, x.size(-2), x.size(-1))
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x

    def glorot_weight_zero_bias(self, model):
        for module in model.modules():
            if hasattr(module, "weight"):
                if not ("BatchNorm" in module.__class__.__name__):
                    init.xavier_uniform_(module.weight, gain=1)
                else:
                    init.constant_(module.weight, 1)
            if hasattr(module, "bias"):
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.temporal_conv(x)
        x = self.graph_embedding(x)
        x = x.view((self.k + 1), batch_size, -1, x.size(-2), x.size(-1))
        x = (
            x.permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(batch_size, -1, x.size(-2), x.size(-1))
        )
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return self.cls(x)
