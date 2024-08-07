# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from collections.abc import Callable

import numpy as np
from numpy import ndarray
from mne.utils import verbose, logger

from .base import Transforms
from ..datasets.base import _DataVar
from ..utils import get_init_args
from .functions import (
    crop,
    slide_win,
    cheby2_filter,
    label_mapping,
    pick_label,
)


class Identity(Transforms):
    """Placeholder identity operator."""

    def __init__(self) -> None:
        super().__init__("Identity()")

    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        return input


class Crop(Transforms):
    """Crop a time interval.

    Parameters
    ----------
    tmin : int
        Start time of selection in sampling points.
    tmax : int
        End time of selection in sampling points. None means use the full time.
    include_tmax : bool
        If `False`, exclude tmax.

    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.
    """

    def __init__(
        self,
        tmin: int = 0,
        tmax: int | None = None,
        include_tmax: bool = False,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.tmin = tmin
        self.tmax = tmax
        self.include_tmax = include_tmax

    @verbose
    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        for eegdata, _ in input.datas():
            eegdata["edata"] = crop(
                eegdata["edata"], self.tmin, self.tmax, self.include_tmax, verbose
            )
        return input


class SlideWin(Transforms):
    """Apply a sliding window to the dataset.

    This transform is only splits the time series (dim = -1) through the
    sliding window operation on the original dataset. If the time axis is
    not divisible by the sliding window, the last remaining time data will
    be discarded.

    Parameters
    ----------
    win : int
        The size of the sliding window.
    overlap : int
        The amount of overlap between adjacent sliding windows.

    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.
    """

    def __init__(self, win: int, overlap: int = 0) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.win = win
        self.overlap = overlap

    @verbose
    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        for eegdata, _ in input.datas():
            eegdata["edata"], eegdata["label"] = slide_win(
                eegdata["edata"], self.win, self.overlap, eegdata["label"], verbose
            )
        return input


class Unsqueeze(Transforms):
    """Insert a dimension on the data.

    This transform is usually used to insert a empty dimension on signals.

    Parameters
    ----------
    key : str
        The key of the eegdata to be transformed.
    dim : int
        Position in the expanded dim where the new dim is placed.

    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.
    """

    def __init__(self, key: str = "edata", dim: int = 1) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.key = key
        self.dim = dim

    @verbose
    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        for eegdata, _ in input.datas():
            eegdata[self.key] = np.expand_dims(eegdata[self.key], self.dim)
        return input


class FilterBank(Transforms):
    """Filter Bank.

    EEG data will be filtered according to different filtering frequencies and
    finally concatenated together. eg.`(Batch, ...) -> (Batch, F, ...)` if the
    number of filter banks exceeds 1, `(Batch, ...) -> (Batch, ...)` if the
    filter has only one. By default, filtering is performed on `edata`, please
    ensure the availability of the data.

    Parameters
    ----------
    freq : float
        Data sampling frequency.
    filter_bank : multiple 2 float of list
        The low-pass and high-pass cutoff frequencies for each filter set.
    transition_bandwidth : float
        The bandwidth (in hertz) of the transition region of the frequency
        response from the passband to the stopband.
    gstop : float
        The minimum attenuation in the stopband (dB).
    gpass : float
        The maximum loss in the passband (dB).

    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.
    """

    def __init__(
        self,
        freq: float,
        filter_bank: list,
        transition_bandwidth: float = 2.0,
        gstop: float = 30,
        gpass: float = 3,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.freq = freq
        self.filter_bank = self._check_filter_bank(filter_bank)
        self.transition_bandwidth = transition_bandwidth
        self.gpass = gpass
        self.gstop = gstop

    def _check_filter_bank(self, fb):
        if not isinstance(fb, list):
            raise TypeError(f"filter_bank must be a list, not {type(fb)}.")
        for f in fb:
            if len(f) != 2:
                raise ValueError(
                    "The filter should be of two variables low pass and high "
                    "pass cutoff frequency."
                )
        return fb

    @verbose
    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        bank_len = len(self.filter_bank)
        for eegdata, _ in input.datas():
            trials = eegdata.trials()
            data = np.empty((trials, bank_len, *eegdata["edata"].shape[1:]))

            for i, cutoff in enumerate(self.filter_bank):
                filter_data = cheby2_filter(
                    data=eegdata["edata"],
                    freq=self.freq,
                    l_freq=cutoff[0],
                    h_freq=cutoff[1],
                    transition_bandwidth=self.transition_bandwidth,
                    gpass=self.gpass,
                    gstop=self.gstop,
                    verbose=verbose,
                )
                data[:, i] = filter_data

            if bank_len == 1:
                data = np.squeeze(data, 1)
            eegdata["edata"] = data

        return input


class ApplyFunc(Transforms):
    """Apply a custom function to data.

    Parameters
    ----------
    func : Callable
        Transformation data callback function. The first parameter of the
        function must be `EEGData`.
    key : str, optional
        The key of the eeg data to be transformed, if required. Applies to all
        eegdata by default.
    **kwargs : dict, optional
        Additional arguments for callback function, if required.

    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.

    Examples
    --------
    If you want to pass a function with parameters, such as you want to use
    `np.expand_dims()` with `axis` parameter, you can do as follows:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> def expand_dim(data, dim=1):
    ...     data["edata"] = np.expand_dims(data["edata"], dim)
    >>> transforms.ApplyFunc(expand_dim, dim=0)(eegdata)
    [edata=(1, 16, 3, 10), label=(16,)]
    >>> split_eegdata = dpeeg.SplitEEGData(eegdata, eegdata.copy())
    >>> transforms.ApplyFunc(expand_dim, "train", dim=3)(split_eegdata)
    {'train': [edata=(1, 16, 3, 1, 10), label=(16,)],
    'test': [edata=(1, 16, 3, 10), label=(16,)]}
    """

    def __init__(
        self,
        func: Callable,
        key: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.func = func
        self.key = key
        self.kwargs = kwargs

    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        for eegdata, key in input.datas():
            if (self.key is not None) and (self.key != key):
                continue
            self.func(eegdata, **self.kwargs)
        return input


class LabelMapping(Transforms):
    """Rearrange the original label according to mapping rules.

    Parameters
    ----------
    mapping : ndarray (2, label_num)
        Label mapping relationship.
    order : bool
        New label start from 0.

    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.
    """

    def __init__(self, mapping: ndarray, order: bool = True) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.mapping = mapping
        self.order = order

    @verbose
    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        for eegdata, _ in input.datas():
            eegdata["label"] = label_mapping(
                eegdata["label"], self.mapping, self.order, verbose
            )
        return input


class PickLabel(Transforms):
    """Pick a subset of data.

    Pick the required labels and data from the dataset and re-label them.

    Parameters
    ----------
    pick : ndarray
        Label to include.


    Returns
    -------
    data : BaseData or BaseDataset
        Transformed eegdata.
    """

    def __init__(self, pick: ndarray) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.pick = pick

    @verbose
    def _apply(self, input: _DataVar, verbose=None) -> _DataVar:
        for eegdata, _ in input.datas():
            eegdata["edata"], eegdata["label"] = pick_label(
                eegdata["edata"], eegdata["label"], self.pick, verbose
            )
        return input
