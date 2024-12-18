# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from collections.abc import Callable

import numpy as np
from numpy import ndarray
from mne.utils import verbose, logger

from .base import Transforms, TransformsEGD
from ..datasets.base import EEGData
from ..utils import DPEEG_SEED, get_init_args
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

    def _apply(self, eegdata):
        return eegdata


class Crop(TransformsEGD):
    """Crop a time interval.

    Crop the eeg signal in terms of time. Default is `edata`.

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
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.Crop(4, 9)(eegdata, verbose=False)
    [edata=(16, 3, 5), label=(16,)]
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

    def _apply_egd(self, egd: EEGData, key: str | None):
        egd["edata"] = crop(
            data=egd["edata"],
            tmin=self.tmin,
            tmax=self.tmax,
            include_tmax=self.include_tmax,
        )


class SlideWin(TransformsEGD):
    """Apply a sliding window to the dataset.

    This transform is only splits the time series (dim = -1) through the
    sliding window operation on the original dataset. If the time axis is
    not divisible by the sliding window, the last remaining time data will
    be discarded. Applied to `edata` and `label` by default.

    Parameters
    ----------
    win : int
        The size of the sliding window (sampling point).
    overlap : int
        The amount of overlap between adjacent windows (sampling point).
    flatten : bool
        If True, return each window as a separate sample. If False, concatenate
        windowed data along one dimension.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.SlideWin(3, 1)(eegdata, verbose=False)
    [edata=(64, 3, 3), label=(64,)]

    Or concatenate the window data as a new sample:

    >>> eegdata = transforms.SlideWin(2, flatten=False)(eegdata, verbose=False)
    [edata=(16, 5, 3, 2), label=(16,)]
    """

    def __init__(self, win: int, overlap: int = 0, flatten: bool = True) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.win = win
        self.overlap = overlap
        self.flatten = flatten

    def _apply_egd(self, egd: EEGData, key: str | None):
        egd["edata"], egd["label"] = slide_win(
            data=egd["edata"],
            win=self.win,
            overlap=self.overlap,
            flatten=self.flatten,
            label=egd["label"],
        )


class Unsqueeze(TransformsEGD):
    """Insert a dimension on the data.

    This transform is usually used to insert a empty dimension on signals.

    Parameters
    ----------
    key : str
        The key of the eegdata value to be transformed.
    dim : int
        Position in the expanded dim where the new dim is placed.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.Unsqueeze(dim=2)(eegdata, verbose=False)
    [edata=(16, 3, 1, 10), label=(16,)]
    """

    def __init__(self, key: str = "edata", dim: int = 1) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.key = key
        self.dim = dim

    def _apply_egd(self, egd: EEGData, key: str | None):
        egd[self.key] = np.expand_dims(egd[self.key], self.dim)


class Squeeze(TransformsEGD):
    """Remove a dimension on the data.

    Parameters
    ----------
    key : str
        The key of the eegdata value to be transformed.
    dim : int
        Selects a subset of the entries of length one in the shape. If a dim is
        selected with shape entry greater than one, an error is raised.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 1, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.Squeeze()(eegdata, verbose=False)
    [edata=(16, 3, 10), label=(16,)]
    """

    def __init__(self, key: str = "edata", dim: int = 1) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.key = key
        self.dim = dim

    def _apply_egd(self, egd: EEGData, key: str | None):
        egd[self.key] = np.squeeze(egd[self.key], self.dim)


class Transpose(TransformsEGD):
    """Data dims transposed.

    By default, the EEG data (``edata``) of eegdata are transposed.

    Parameters
    ----------
    dims : tuple or list of int, optinal
        A tuple or list contains a permutation of [0,1,...,N-1] where N is the
        number of dims of the key values. The ``i`` ^th dim of the value will
        correspond to the axis numbered ``dims[i]`` of the input. If not
        specified, reverse the dims order by default.
    key : str
        The key of the eegdata value to be transformed.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 1, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.Transpose()(eegdata, verbose=False)
    [edata=(10, 3, 16), label=(16,)]
    """

    def __init__(self, dims: list[int] | None = None, key: str = "edata"):
        super().__init__(get_init_args(self, locals(), "rp"))
        self.dims = dims
        self.key = key

    def _apply_egd(self, egd: EEGData, key: str | None):
        egd[self.key] = np.transpose(egd[self.key], self.dims)


class FilterBank(TransformsEGD):
    """Filter Bank.

    EEG data will be filtered according to different filtering frequencies and
    finally concatenated together. eg.`(Batch, ...) -> (Batch, F, ...)` if the
    number of filter banks exceeds 1, `(Batch, ...) -> (Batch, ...)` if the
    filter has only one. By default, filtering is performed on `edata`, please
    ensure the availability of the data. Related references include [1]_ and
    [2]_.

    Parameters
    ----------
    freq : float
        EEG data sampling frequency.
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
    data : eegdata or dataset
        Transformed eegdata.

    References
    ----------
    .. [1] R. Mane, E. Chew, K. Chua, K. K. Ang, N. Robinson, A. P. Vinod,
        S.-W. Lee, and C. Guan, “FBCNet: A multi-view convolutional neural
        network for brain-computer interface,” arXiv preprint arXiv:2104.01233,
        2021.
    .. [2] X. Ma, W. Chen, Z. Pei, J. Liu, B. Huang, and J. Chen, “A temporal
        dependency learning CNN with attention mechanism for MI-EEG decoding,”
        IEEE Transactions on Neural Systems and Rehabilitation Engineering,
        2023.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.FilterBank(250)(eegdata, verbose=False)
    [edata=(16, 9, 3, 10), label=(16,)]
    """

    def __init__(
        self,
        freq: float,
        filter_bank: list = [
            [4, 8],
            [8, 12],
            [12, 16],
            [16, 20],
            [20, 24],
            [24, 28],
            [28, 32],
            [32, 36],
            [36, 40],
        ],
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
        self.bank_len = len(self.filter_bank)

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

    def _apply_egd(self, egd: EEGData, key: str | None):
        trials = egd.trials()
        data = np.empty((trials, self.bank_len, *egd["edata"].shape[1:]))

        for i, cutoff in enumerate(self.filter_bank):
            filter_data = cheby2_filter(
                data=egd["edata"],
                freq=self.freq,
                l_freq=cutoff[0],
                h_freq=cutoff[1],
                transition_bandwidth=self.transition_bandwidth,
                gpass=self.gpass,
                gstop=self.gstop,
            )
            data[:, i] = filter_data

        if self.bank_len == 1:
            data = np.squeeze(data, 1)
        egd["edata"] = data


class ApplyFunc(TransformsEGD):
    """Apply a custom function to data.

    Parameters
    ----------
    func : Callable
        Transformation data callback function. The first parameter of the
        function must be `EEGData`.
    keys : list of str, optional
        The key of the eegdata to be transformed, if required. Applies to all
        eegdata by default.
    **kwargs : dict, optional
        Additional arguments for callback function, if required.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    If you want to pass a function with parameters, such as you want to use
    `np.expand_dims()` with `axis` parameter, you can do as follows:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> def expand_dim(data, dim=1):
    ...     data["edata"] = np.expand_dims(data["edata"], dim)
    >>> transforms.ApplyFunc(expand_dim, dim=0)(eegdata, verbose=False)
    [edata=(1, 16, 3, 10), label=(16,)]

    >>> split_eegdata = dpeeg.SplitEEGData(eegdata, eegdata.copy())
    >>> transforms.ApplyFunc(expand_dim, ["train"])(split_eegdata, verbose=False)
    Train: [edata=(1, 1, 16, 3, 10), label=(16,)]
    Test : [edata=(1, 16, 3, 10), label=(16,)]
    """

    def __init__(
        self,
        func: Callable,
        keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.func = func
        self.keys = keys
        self.kwargs = kwargs

    def _apply_egd(self, egd: EEGData, key: str | None):
        if (self.keys is None) or (key in self.keys):
            self.func(egd, **self.kwargs)


class LabelMapping(TransformsEGD):
    """Update the original label according to mapping rules.

    Parameters
    ----------
    mapping : ndarray (2, label_num), optional
        Label mapping relationship. The first row is the original label, and
        the second row is the mapped label. If ``None``, the label will be
        updated in ascending order starting from zero.
    order : bool
        Force the new labels to start incrementing from zero.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> eegdata['label']
    array([3, 2, 2, 2, 3, 2, 4, 3, 4, 3, 3, 2, 4, 4, 2, 3])

    Merge labels as needed:

    >>> transforms.LabelMapping(
    ...     np.array([[2, 3, 4], [0, 0, 1]])
    ... )(eegdata, verbose=False)
    >>> eegdata["label"]
    array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0])
    """

    def __init__(self, mapping: ndarray | None = None, order: bool = True) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.mapping = mapping
        self.order = order

    def _apply_egd(self, egd: EEGData, key: str | None):
        egd["label"] = label_mapping(
            label=egd["label"],
            mapping=self.mapping,
            order=self.order,
        )


class PickLabel(TransformsEGD):
    """Pick a subset of data.

    Pick the required labels and data from the dataset and re-label them.

    Parameters
    ----------
    pick : list of int
        Label to include.
    keys : list of str, optional
        The key of the eegdata value to be transformed, if required. Applies to
        all eegdata by default.
    order : bool
        If `True`, relabel the selected labels.
    shuffle : bool
        Whether or not to shuffle the data after picking.
    seed : int
        Controls the shuffling applied to the data after picking.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    array([1, 2, 0, 2, 1, 2, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0])

    >>> transforms.PickLabel(np.array([1, 2]))(eegdata, verbose=False)
    array([1, 0, 1, 0, 1, 0, 0, 0, 1])

    If some values do not need to be transformed, they can be excluded by the
    `keys` parameter:

    >>> eegdata = dpeeg.EEGData(
    ...     edata=np.random.randn(16, 3, 10),
    ...     label=np.random.randint(0, 3, 16),
    ...     adj=np.random.randn(16, 3, 3),
    ...     pcc=np.random.randn(16, 3, 3),
    ... )
    >>> transforms.PickLabel(
    ...    np.array([0, 1]), keys=["edata", "adj"]
    ... )(eegdata, verbose=False)
    [edata=(12, 3, 10), label=(12,), adj=(12, 3, 3), pcc=(16, 3, 3)]
    """

    def __init__(
        self,
        pick: list[int],
        keys: list[str] | None = None,
        order: bool = True,
        shuffle: bool = True,
        seed: int = DPEEG_SEED,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        self.pick = pick
        self.keys = keys
        self.order = order
        self.shuffle = shuffle
        self.seed = seed

    def _apply_egd(self, egd: EEGData, key: str | None):
        label = egd["label"]

        keys, values = [], []
        for key, value in egd.items():
            if (key != "label") and ((self.keys is None) or (key in self.keys)):
                keys.append(key)
                values.append(value)

        data, label = pick_label(
            *values,
            label=label,
            pick=self.pick,
            order=self.order,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        egd["label"] = label
        for i, key in enumerate(keys):
            egd[key] = data[i]
