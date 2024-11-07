# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import abstractmethod

from numpy import ndarray
from mne.utils import verbose, logger

from .base import Transforms
from ..datasets.base import EEGData, _BaseData, SplitEEGData
from ..utils import get_init_args
from ..tools.docs import fill_doc
from .functions import (
    z_score_norm,
    min_max_norm,
)


class Normalization(Transforms):
    """Data normalization base class.

    Normalize the data, with default normalization applied to the `edata`.
    Please verify the validity of the data.

    Attributes
    ----------
    params : dict
        Saves intermediate variables of calculations.
    """

    def __init__(self, repr: str | None = None, train_for_test: bool = False) -> None:
        super().__init__(repr)
        self.train_for_test = train_for_test
        self.params = {}

    def _apply(self, eegdata: _BaseData) -> _BaseData:
        if (not isinstance(eegdata, SplitEEGData)) and self.train_for_test:
            raise TypeError(
                "The input must have been split when `train_for_test` is True."
            )
        else:
            for egd, mode in eegdata._datas():
                self._apply_norm(egd, mode)

        return eegdata

    @abstractmethod
    def _apply_norm(self, egd: EEGData, mode: str):
        pass


@fill_doc
class ZscoreNorm(Normalization):
    r"""Z-score normalization per subject.

    By default, the EEG data (``edata``) of eegdata are normalized.

    .. math::

       \mathbf{z} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^{2}}}

    where :math:`\mathbf{x}` and :math:`\mathbf{z}` denote the input data and
    the output of normalization, respectively. :math:`\mu` and :math:`\sigma^2`
    represent the mean and variance values of the sample.

    Parameters
    ----------
    %(train_for_test)s
    mean : ndarray, optional
        The mean used in the normalization process. If None, use the statistics
        of the current sample for normalization. Ignored when ``train_for_test``
        is True.
    std : ndarray, optional
        The standard deviation used in the normalization process. If None, use
        the statistics of the current sample for normalization. Ignored when
        ``train_for_test`` is True.
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
        train_for_test: bool = False,
        mean: ndarray | None = None,
        std: ndarray | None = None,
        dim: int | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), format="rp"),
            train_for_test=train_for_test,
        )
        self.mean = mean
        self.std = std
        self.dim = dim

    def _apply_norm(self, egd: EEGData, mode: str):
        if self.train_for_test:
            if mode == "train":
                egd["edata"], mean, std = z_score_norm(
                    egd["edata"], dim=self.dim, ret=True
                )
                self.params[self.subject] = {"mean": mean, "std": std}
            else:
                egd["edata"] = z_score_norm(
                    egd["edata"],
                    mean=self.params[self.subject]["mean"],
                    std=self.params[self.subject]["std"],
                    dim=self.dim,
                    ret=False,
                )
        else:
            egd["edata"], mean, std = z_score_norm(
                egd["edata"], self.mean, self.std, self.dim, True
            )
            self.params.update({"mean": mean, "std": std})


@fill_doc
class MinMaxNorm(Normalization):
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
    %(train_for_test)s
    min : ndarray, optional
        The minimum used in the normalization process. If None, use the
        statistics of the current sample for normalization. Ignored when
        ``train_for_test`` is True.
    max : ndarray, optional
        The maximum used in the normalization process. If None, use the
        statistics of the current smaple for normalization. Ignored when
        ``train_for_test`` is True.
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
        train_for_test: bool = False,
        min: ndarray | None = None,
        max: ndarray | None = None,
        dim: int | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), format="rp"),
            train_for_test=train_for_test,
        )
        self.train_for_test = train_for_test
        self.min = min
        self.max = max
        self.dim = dim

    def _apply_norm(self, egd: EEGData, mode: str):
        if self.train_for_test:
            if mode == "train":
                egd["edata"], min, max = min_max_norm(
                    egd["edata"], dim=self.dim, ret=True
                )
                self.params[self.subject] = {"min": min, "max": max}
            else:
                egd["edata"] = min_max_norm(
                    egd["edata"],
                    min=self.params[self.subject]["min"],
                    max=self.params[self.subject]["max"],
                    dim=self.dim,
                    ret=False,
                )
        else:
            egd["edata"], min, max = min_max_norm(
                egd["edata"], self.min, self.max, self.dim, True
            )
            self.params.update({"min": min, "max": max})
