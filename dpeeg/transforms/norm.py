# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from numpy import ndarray
from mne.utils import verbose, logger

from .base import Transforms
from ..datasets.base import _DataVar
from ..utils import get_init_args
from .functions import (
    z_score_norm,
    min_max_norm,
)


class ZscoreNorm(Transforms):
    r"""Z-score normalization per subject.

    By default, the EEG data (``edata``) of eegdata are normalized.

    .. math::

       \mathbf{z} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^{2}}}

    where :math:`\mathbf{x}` and :math:`\mathbf{z}` denote the input data and
    the output of normalization, respectively. :math:`\mu` and :math:`\sigma^2`
    represent the mean and variance values of the sample.

    Parameters
    ----------
    mean : ndarray, optional
        The mean used in the normalization process. If None, use the statistics
        of the current sample for normalization.
    std : ndarray, optional
        The standard deviation used in the normalization process. If None, use
        the statistics of the current sample for normalization.
    dim : int, optional
        The dimension to normalize. Usually, -1 for channels and -2 for time
        points. If None, normalize at the sample level.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.
    """

    def __init__(
        self,
        mean: ndarray | None = None,
        std: ndarray | None = None,
        dim: int | None = None,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.mean = mean
        self.std = std
        self.dim = dim

    @verbose
    def _apply(self, eegdata: _DataVar, verbose=None) -> _DataVar:
        logger.info(f"  Apply {self} ...")

        for egd, _ in eegdata._datas():
            egd["edata"] = z_score_norm(
                egd["edata"], self.mean, self.std, self.dim, verbose=verbose
            )
        return eegdata


class MinMaxNorm(Transforms):
    r"""Min-max normalization per subject.

    By default, the EEG data (``edata``) of eegdata are normalized.

    .. math::

       \mathbf{z} =
       \frac{\mathbf{x}-\mathbf{x}_{min}}{\mathbf{x}_{max}-\mathbf{x}_{min}}

    where :math:`\mathbf{x}` and :math:`\mathbf{z}` denote the input data and
    the output of normalization, respectively. :math:`\mathbf{x}_{max}` and
    :math:`\mathbf{x}_{min}` represent the maximum and minimum values of the
    sample.

    Parameters
    ----------
    min : ndarray, optional
        The minimum used in the normalization process. If None, use the
        statistics of the current sample for normalization.
    max : ndarray, optional
        The maximum used in the normalization process. If None, use the
        statistics of the current smaple for normalization.
    dim : int, optional
        The dimension to normalize. Usually, -1 for channels and -2 for time
        points. If None, normalize at the sample level.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.
    """

    def __init__(
        self,
        min: ndarray | None = None,
        max: ndarray | None = None,
        dim: int | None = None,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.min = min
        self.max = max
        self.dim = dim

    @verbose
    def _apply(self, eegdata: _DataVar, verbose=None) -> _DataVar:
        logger.info(f"  Apply {self} ...")

        for egd, _ in eegdata._datas():
            egd["edata"] = min_max_norm(
                egd["edata"], self.min, self.max, self.dim, verbose=verbose
            )
        return eegdata
