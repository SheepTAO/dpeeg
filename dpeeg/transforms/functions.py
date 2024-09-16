# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from typing import Literal

import torch
import numpy as np
import scipy.signal as signal
from numpy import ndarray
from mne.utils import verbose, logger
from torch import Tensor

from ..utils import DPEEG_SEED, iterable_to_str


@verbose
def split_train_test(
    *arrs,
    test_size: float = 0.25,
    seed: int = DPEEG_SEED,
    sample: list[int] | None = None,
    verbose=None,
) -> list:
    """Split an dataset into training and testing sets. The axis along which
    to split is 0.

    .. deprecated:: 0.4.0
        Will be removed in a later version.

    Parameters
    ----------
    *arrs : sequence of indexables with same length / shape[0]
        Allowed inputs are lists and numpy arrays.
    test_size : float
        The proportion of the test set. If index is not None, test_size will be
        ignored. Default use stratified fashion and the last arr serves as the
        class label.
    seed : int
        Random seed when splitting.
    sample : list of int, None
        A list of integers, the entries indicate which data were selected
        as the test set. If None, test_size will be used.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    num_arrs = len(arrs)
    if num_arrs == 0:
        raise ValueError("At least one array required as input.")

    arr_list = [np.array(arr) for arr in arrs]
    lengths = [len(arr) for arr in arr_list]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(l) for l in lengths]
        )

    if sample is not None:
        smparr = np.array(sample)
        if np.unique(smparr).size != smparr.size:
            raise IndexError(f"Found repeated sampling of test set: {smparr.tolist()}.")
        length = lengths[0]
        if smparr.size >= uniques:
            raise IndexError(
                f"The number of samples (={len(smparr)}) in the "
                "test set cannot exceed the total number of data "
                f"sets (={length})."
            )

        testSample, trainSample = smparr, np.setdiff1d(np.arange(length), smparr)
        res = []
        for arr in arr_list:
            res.extend([arr[trainSample], arr[testSample]])
        return res
    else:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            *arr_list, test_size=test_size, random_state=seed, stratify=arr_list[-1]
        )


@verbose
def merge_train_test(*arrs, verbose=None) -> tuple[ndarray, ndarray]:
    """Merge the data and label of the training set and test set.

    .. deprecated:: 0.4.0
        Will be removed in a later version.

    Parameters
    ----------
    *arrs : sequence of data (ndarray (N, ...)) and label (ndarray (N,))
        Sequence consisting of each piece of data and its corresponding label.

    Returns
    -------
    data, label : ndarray, ndarray
        Merged data and label.

    Examples
    --------
    >>> data = np.arange(6).reshape(2, 3)
    >>> label = np.arange(2)
    >>> train, test = [data, label], [data, label]
    >>> merge_train_test(train, test)
    (array([[3, 4, 5],
            [0, 1, 2],
            [0, 1, 2],
            [3, 4, 5]]),
    array([1, 0, 0, 1]))
    """
    if len(arrs) == 0:
        raise ValueError("At least one array required as input.")

    merge_data = []
    merge_label = []
    for i, arr in enumerate(arrs):
        if not (isinstance(arr, (list, tuple)) and len(arr) == 2):
            raise ValueError(f"Missing data or label in {i}^th group.")

        data, label = arr[0], arr[1]
        if data.shape[0] != label.shape[0]:
            logger.warning(f"Length of data and label in {i}^th group is not uniform.")
        merge_data.append(data)
        merge_label.append(label)

    merge_data = np.concatenate(merge_data)
    merge_label = np.concatenate(merge_label)
    shuffle = np.random.permutation(merge_label.shape[0])
    merge_data = merge_data[shuffle]
    merge_label = merge_label[shuffle]

    return merge_data, merge_label


@verbose
def to_tensor(
    data: Tensor | ndarray, label: Tensor | ndarray, verbose=None
) -> tuple[Tensor, Tensor]:
    """Convert the numpy data and label into trainable Tensor format."""
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    dataT = (
        torch.from_numpy(np.ascontiguousarray(data)).float()
        if isinstance(data, ndarray)
        else data.float()
    )
    labelT = (
        torch.from_numpy(np.ascontiguousarray(label)).long()
        if isinstance(label, ndarray)
        else label.long()
    )
    return dataT, labelT


@verbose
def crop(
    data: ndarray,
    tmin: int = 0,
    tmax: int | None = None,
    include_tmax: bool = False,
    verbose=None,
) -> ndarray:
    """Crop a time interval from the data along the last dimension by default.

    Parameters
    ----------
    tmin : int
        Start time of selection in sampling points.
    tmax : int, None
        End time of selection in sampling points. None means use the full time.
    include_tmax : bool
        If `False`, exclude tmax.

    Returns
    -------
    data : ndarray
        The cropped data.
    """
    if tmax is not None and include_tmax:
        tmax += 1
        assert tmax > tmin, "tmax must be greater than tmin"
    return data[..., tmin:tmax]


@verbose
def z_score_norm(
    data: ndarray,
    mean: ndarray | None = None,
    std: ndarray | None = None,
    dim: int | None = None,
    verbose=None,
) -> ndarray | tuple[ndarray, ndarray]:
    r"""Z-score normalization.

    .. math::

       \mathbf{z} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^{2}}}

    where :math:`\mathbf{x}` and :math:`\mathbf{z}` denote the input data and
    the output of normalization, respectively. :math:`\mu` and :math:`\sigma^2`
    represent the mean and variance values of the sample.

    Parameters
    ----------
    data : ndarray
        Input data.
    mean : ndarray, optional
        The mean used in the normalization process. If None, use the statistics
        of the current sample for normalization.
    std : ndarray, optional
        The standard deviation used in the normalization process. If None, use
        the statistics of the current sample for normalization.
    dim : int, optional
        The dimension to normalize. If None, the entire data is normalized.

    Returns
    -------
    ndarray
        Normalized data.
    """
    keepdims = False if dim is None else True
    mean = np.mean(data, dim, keepdims=keepdims) if mean is None else mean
    std = np.std(data, dim, keepdims=keepdims) if std is None else std
    return (data - mean) / std


@verbose
def min_max_norm(
    data: ndarray,
    min: ndarray | None = None,
    max: ndarray | None = None,
    dim: int | None = None,
    verbose=None,
) -> ndarray:
    r"""Min-max normalization.

    .. math::

       \mathbf{z} =
       \frac{\mathbf{x}-\mathbf{x}_{min}}{\mathbf{x}_{max}-\mathbf{x}_{min}}

    where :math:`\mathbf{x}` and :math:`\mathbf{z}` denote the input data and
    the output of normalization, respectively. :math:`\mathbf{x}_{max}` and
    :math:`\mathbf{x}_{min}` represent the maximum and minimum values of the
    sample.

    Parameters
    ----------
    data : ndarray
        Input data.
    min : ndarray, optional
        The minimum used in the normalization process. If None, use the
        statistics of the current sample for normalization.
    max : ndarray, optional
        The maximum used in the normalization process. If None, use the
        statistics of the current smaple for normalization.
    dim : int, None
        The dimension to normalize. If None, the entire data is normalized.

    Returns
    -------
    ndarray
        Normalized data.
    """
    keepdims = False if dim is None else True
    minimum = np.min(data, dim, keepdims=keepdims) if min is None else min
    maximum = np.max(data, dim, keepdims=keepdims) if max is None else max
    return (data - minimum) / (maximum - minimum)


@verbose
def slide_win(
    data: ndarray,
    win: int,
    overlap: int = 0,
    label: ndarray | None = None,
    verbose=None,
) -> tuple[ndarray, ndarray | None]:
    """This transform is only splits the time series (dim = -1) through the
    sliding window operation on the original dataset. If the time axis is not
    divisible by the sliding window, the last remaining time data will be
    discarded.

    Parameters
    ----------
    data : ndarray (N, ..., T)
        The data to split. Shape as `(N, ..., T)`, with `N` the number of data
        and `T` the number of samples.
    win : int
        The size of the sliding window.
    overlap : int
        The amount of overlap between adjacent sliding windows.
    label : ndarray (N,), None
        The label of the data. If not None, label will update with sliding window.

    Returns
    -------
    data : ndarray
        Data after sliding window.
    label : ndarray, None
        If label is None, return None. The label corresponding to each window.
    """
    if win < 0 or overlap < 0:
        raise ValueError(
            "win and overlap only allow positive numbers, "
            f"but got {win} and {overlap} respectively."
        )

    if overlap >= win:
        raise ValueError(f"overlap={overlap} should be less than win={win}.")

    if isinstance(label, ndarray) and len(data) != len(label):
        raise ValueError("The number of label and data must be the same.")

    end = win
    times = data.shape[-1]
    if end > times:
        logger.warning("The window is larger than the times to be split.")
        return data, label

    sld_num = 0
    data_list = []
    while end <= times:
        data_list.append(data[..., end - win : end])
        sld_num += 1
        end += win - overlap

    data = np.concatenate(data_list)
    shuffle = np.random.permutation(len(data))
    data = data[shuffle]
    if isinstance(label, ndarray):
        label = np.repeat(label, sld_num)
        label = label[shuffle]
    return data, label


@verbose
def segmentation_and_reconstruction_time(
    data: ndarray,
    label: ndarray,
    samples: int = 125,
    multiply: float = 1.0,
    shuffle: bool = True,
    seed: int = DPEEG_SEED,
    verbose=None,
) -> tuple[ndarray, ndarray]:
    """Signal Segmentation and Recombination in Time Domain.

    This approach is to first divide each EEG trial into several segments and
    then generate new artificial trials as a concatenation of segments coming
    from different and randomly selected trials from the same class while
    maintaining the original time order [1]_.

    Parameters
    ----------
    data : ndarray (N, ..., T)
        Data that will be segmented and randomly recombined. Shape as `(N, ...,
        T)`, with `N` the number of data and `T` the number of samples.
    label : ndarray (N,)
        The label corresponding to the data. Shape as `(N)`.
    samples : int
        The number of consecutive samples to segment the data. eg, 125 for
        250Hz data is segmented by 0.5s.
    multiply : float
        Data expansion multiple of relative metadata, 1 means doubled.
    shuffle : bool
        Whether or not to shuffle the data after picking.
    seed : int
        Controls the shuffling applied to the data after picking.

    Notes
    -----
    The function does not judge the multiples or the number of segmentation.

    References
    ----------
    .. [1] F. Lotte, “Signal Processing Approaches to Minimize or Suppress
        Calibration Time in Oscillatory Activity-Based Brain-Computer
        Interfaces,” Proc. IEEE, vol. 103, no. 6, pp. 871-890, Jun. 2015,
        doi: 10.1109/JPROC.2015.2404941.
    """
    assert samples >= 1, "samples should be at least 1"
    assert multiply > 0, "multiply should be greater than 0"

    parts = data.shape[-1] // samples
    aug_data, aug_label = [], []

    for lb in np.unique(label):
        idx = np.where(label == lb)
        org_data, org_label = data[idx], label[idx]
        aug_num = int(org_data.shape[0] * multiply)
        tmp_aug_data = np.empty((aug_num, *data.shape[1:]))

        for i in range(aug_num):
            for j in range(parts):
                randIdx = np.random.randint(0, org_data.shape[0], parts)
                tmp_aug_data[i, ..., j * samples : (j + 1) * samples] = org_data[
                    randIdx[j], ..., j * samples : (j + 1) * samples
                ]
            if data.shape[-1] % samples:
                randIdx = np.random.randint(0, org_data.shape[0])
                tmp_aug_data[i, ..., (j + 1) * samples :] = org_data[
                    randIdx, ..., (j + 1) * samples :
                ]

        aug_data.append(tmp_aug_data)
        aug_label.append(np.repeat(lb, aug_num))
        aug_data.append(org_data)
        aug_label.append(org_label)

    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)

    if shuffle:
        rng = np.random.RandomState(seed)
        shuffle_idx = rng.permutation(aug_label.shape[0])
        aug_data = aug_data[shuffle_idx]
        aug_label = aug_label[shuffle_idx]

    return aug_data, aug_label


@verbose
def cheby2_filter(
    data: ndarray,
    freq: float,
    l_freq: float | None = None,
    h_freq: float | None = None,
    transition_bandwidth: float = 2.0,
    gstop: float = 30,
    gpass: float = 3,
    filter_type: Literal["filter", "filtfilt"] = "filter",
    verbose=None,
):
    """Filter a signal using cheby2 iir filtering.

    Parameters
    ----------
    data : ndarray
        Filtering is performed in the last dimension (time dimension).
    freq : float
        Data sampling frequency.
    l_freq, h_freq : float, None
        Low and high cut off frequency in hertz. If l_freq is None, the data
        are only low-passed. If h_freq is None,  the data are only high-passed.
        Both parameters cannot be None at the same time.
    transition_bandwidth : float
        The bandwidth (in hertz) of the transition region of the frequency res-
        ponse from the passband to the stopband.
    gstop : float
        The minimum attenuation in the stopband (dB).
    gpass : float
        The maximum loss in the passband (dB).
    filter_type : str
        Filter type, available options are 'filtfilt' and 'filter'.

    Returns
    -------
    data : ndarray
        Data after applying bandpass filter.
    """

    n_freq = freq / 2  # Nyquist frequency

    if (l_freq == 0 or l_freq == None) and (h_freq == None or h_freq >= n_freq):
        logger.warning("Not doing any filtering. Invalid cut-off freq.")
        return data

    # low-passed filter
    elif l_freq == 0 or l_freq == None:
        logger.info(f"{h_freq} Hz lowpass filter.")
        fpass = h_freq / n_freq  # type:ignore
        fstop = (h_freq + transition_bandwidth) / n_freq  # type:ignore
        N, Ws = signal.cheb2ord(fpass, fstop, gpass, gstop)
        b, a = signal.cheby2(N, gstop, fstop, "lowpass")

    # high-passed filter
    elif h_freq == None or h_freq == n_freq:
        logger.info(f"{l_freq} Hz highpass filter.")
        fpass = l_freq / n_freq
        fstop = (l_freq - transition_bandwidth) / n_freq
        N, Ws = signal.cheb2ord(fpass, fstop, gpass, gstop)
        b, a = signal.cheby2(N, gstop, fstop, "highpass")

    # band-passed filter
    else:
        logger.info(f"{l_freq} - {h_freq} Hz bandpass filter.")
        fpass = [l_freq / n_freq, h_freq / n_freq]
        fstop = [
            (l_freq - transition_bandwidth) / n_freq,
            (h_freq + transition_bandwidth) / n_freq,
        ]
        N, Ws = signal.cheb2ord(fpass, fstop, gpass, gstop)
        b, a = signal.cheby2(N, gstop, fstop, "bandpass")

    if filter_type == "filtfilt":
        out = signal.filtfilt(b, a, data)
    else:
        out = signal.lfilter(b, a, data)

    return out


@verbose
def label_mapping(
    label: ndarray,
    mapping: ndarray | None = None,
    order: bool = True,
    verbose=None,
) -> ndarray:
    """Rearrange the original label according to mapping rules.

    Parameters
    ----------
    label : ndarray (N,)
        Original label list.
    mapping : ndarray (2, label_num)
        Label mapping relationship. The first row is the original label, and
        the second row is the mapped label. If ``None``, the label will be
        reordered in ascending order starting from 0.
    order : bool
        Force the new labels to start incrementing from 0.

    Returns
    -------
    Returns the mapped label.

    Examples
    --------
    Merge label:

    >>> label = np.array([1, 2, 3, 2, 3, 1, 3, 4])
    >>> mapping = np.array([[1, 2, 3, 4], [0, 1, 0, 1]])
    >>> label_mapping(label, mapping)
    array([0, 1, 0, 1, 0, 0, 0, 1])

    Rearrange the original label:

    >>> mapping = np.array([[1, 2, 3, 4], [3, 2, 1, 0]])
    >>> label_mapping(label, mapping)
    array([3, 2, 1, 2, 1, 3, 1, 0])
    """
    label, uni_label = np.array(label), np.unique(label)

    if mapping is None:
        mapping = np.array([uni_label, np.arange(len(uni_label))])
        order = False
    else:
        mapping, uni_mapping = np.array(mapping), np.unique(mapping[0])

        if mapping.ndim != 2 or mapping.shape[0] != 2:
            raise ValueError("The mapping is not 2D.")

        if len(uni_label) != len(uni_mapping) or any(uni_label != uni_mapping):
            raise ValueError("Mapping does not correspond to label.")

    if order:
        uni_new_label = np.unique(mapping[1])
        order_label = np.arange(len(uni_new_label))
        if any(order_label != uni_new_label):
            raise ValueError("Mapping error, set `order = false` to turn off.")

    new_label = np.empty_like(label)
    for i in range(mapping.shape[1]):
        indices = np.where(mapping[0][i] == label)
        new_label[indices] = mapping[1][i]

    return new_label


@verbose
def pick_label(
    *data: ndarray,
    label: ndarray,
    pick: ndarray,
    order: bool = False,
    shuffle: bool = True,
    seed: int = DPEEG_SEED,
    verbose=None,
):
    """Pick a subset of data by label.

    Pick the required labels and data from the dataset or re-label them.

    Parameters
    ----------
    *data : sequence of indexables with same length / shape[0]
        The data to pick. Shape as `(N, ...)`.
    label : ndarray (N,)
        Dataset label.
    pick : ndarray (n,)
        Label to include.
    order : bool
        If `True`, relabel the selected labels.
    shuffle : bool
        Whether or not to shuffle the data after picking.
    seed : int
        Controls the shuffling applied to the data after picking.

    Returns
    -------
    data, label : list of ndarray, length = len(data) + 1
        Returns the picked data and label.

    Examples
    --------
    >>> data1 = np.arange(24).reshape(8, 3)
    >>> data2 = np.arange(2, 26).reshape(8, 3)
    >>> label = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    >>> pick_label(data1, data2, label=label, pick=np.array([1]))
    ([array([[ 9, 10, 11],
             [ 6,  7,  8]]),
      array([[11, 12, 13],
             [ 8,  9, 10]])],
     array([1, 1]))

    >>> pick_label(data1, label=label, pick=np.array([0, 2]), order=True)
    ([array([[ 0,  1,  2],
             [ 3,  4,  5],
             [21, 22, 23],
             [12, 13, 14],
             [18, 19, 20],
             [15, 16, 17]])],
     array([0, 0, 1, 1, 1, 1]))
    """
    num_data = len(data)
    if num_data == 0:
        raise ValueError("At least one data required as input.")

    elements = set(pick) - set(label)
    if elements:
        raise ValueError(f"{iterable_to_str(elements)} not in label.")

    new_label = []
    new_data = [[] for _ in range(num_data)]
    for i, p in enumerate(np.unique(pick)):
        indices = np.where(label == p)[0]
        for n in range(num_data):
            new_data[n].append(data[n][indices])

        if order:
            new_label.append(np.repeat(i, len(indices)))
        else:
            new_label.append(label[indices])

    new_data = [np.concatenate(nd) for nd in new_data]
    new_label = np.concatenate(new_label)

    if shuffle:
        rng = np.random.RandomState(seed)
        shuffle_idx = rng.permutation(new_label.shape[0])
        new_data = [nd[shuffle_idx] for nd in new_data]
        new_label = new_label[shuffle_idx]

    return new_data, new_label


def pick_data(data: ndarray, labels: ndarray, label: int) -> ndarray:
    """Index data based on specified label.

    Parameters
    ----------
    data : ndarray (N, ...)
        Input data.
    labels : ndarray (N,)
        Label table corresponding to the data.
    label : int
        The label to get.

    Returns
    -------
    Tensor
        All data for the specified label.
    """
    assert label in labels, f"label {label} is not in the labels."

    indices = np.where(labels == label)
    return data[indices]


def _NPMConv(a: ndarray, v: ndarray, mode: Literal["full", "same", "valid"]) -> ndarray:
    """Linear convolution of two multi-dimensional sequences."""
    dim, seq = a.shape[:-1], a.shape[-1]
    a_flatten = a.reshape(-1, seq)
    for i in range(len(a_flatten)):
        a_flatten[i] = np.convolve(a_flatten[i], v, mode)
    return a_flatten.reshape(*dim, seq)


def smooth(signal: ndarray, win: int = 5) -> ndarray:
    """Matlab smooth implement for moving average method in python.

    Parameters
    ----------
    signal : ndarray (N,)
        Input signal.
    win : int
        Dimension of the smoothing window. If you specify win as an even number,
        win is automatically reduced by 1.
    """
    sig_len = len(signal)
    win = win if win % 2 else win - 1
    assert win < sig_len, f"window:{win} exceeds signal length:{sig_len}."

    sig_mid = np.convolve(signal, np.ones(win) / win, mode="valid")
    r = np.arange(1, win - 1, 2)
    sig_beg = np.cumsum(signal[: win - 1])[::2] / r
    sig_end = (np.cumsum(signal[:-win:-1])[::2] / r)[::-1]
    return np.concatenate((sig_beg, sig_mid, sig_end))


def erds_time(
    data: ndarray,
    ref_start: int,
    ref_end: int,
    smooth_win: int,
) -> tuple[ndarray, ndarray]:
    """Time course of ERD/ERS.

    Returns
    -------
    erds : ndarray
        ERD/ERS in %.
    avg_power : ndarray
        Averaged power.
    """
    assert ref_start >= 0, "ref_start is less than 0."
    assert ref_end < data.shape[-1], "ref_end exceeds data time range."
    assert ref_end > ref_start, "ref_end is less than ref_start."

    avg_power = smooth(data.var(axis=0, ddof=1), smooth_win)
    ref_avg_power = np.mean(avg_power[ref_start:ref_end])
    erds = ((avg_power - ref_avg_power) / ref_avg_power) * 100

    return erds, avg_power
