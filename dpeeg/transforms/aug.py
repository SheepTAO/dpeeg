# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import abstractmethod

from mne.utils import verbose, logger

from ..datasets.base import _BaseData, EEGData, SplitEEGData
from .base import Transforms
from ..utils import DPEEG_SEED, get_init_args
from .functions import segmentation_and_reconstruction_time, slide_win, crop


class Augmentation(Transforms):
    """Data augmentation base class.

    Augment the data, with default augmentation applied to the `edata` and
    `label`. Please verify the validity of the data.

    Parameters
    ----------
    only_train : bool
        If True, data augmentation is performed only on the training set.
    strict : bool
        If False, allow the input data to be unsplit. In this case, data
        augmentation will be applied to all data.
    """

    def __init__(
        self, repr: str | None = None, only_train: bool = True, strict: bool = True
    ) -> None:
        super().__init__(repr)
        self.only_train = only_train
        self.strict = strict

    @verbose
    def _apply(self, eegdata: _BaseData, verbose=None) -> _BaseData:
        logger.info(f"  Apply {self} ...")

        if not isinstance(eegdata, SplitEEGData):
            if self.strict:
                raise TypeError(
                    "The input must have been split, or `strict` is set to False."
                )
            else:
                for egd, _ in eegdata._datas():
                    self._apply_aug(egd, "None", verbose)
        else:
            for egd, mode in eegdata._datas():
                if (mode != "train") and self.only_train:
                    continue
                self._apply_aug(egd, mode, verbose)

        return eegdata

    @abstractmethod
    def _apply_aug(self, eegdata: EEGData, mode: str, verbose=None) -> EEGData:
        pass


class SegRecTime(Augmentation):
    """Segmentation and reorganization in the time domain.

    The S&R process involves segmenting the original eeg signals based on class
    along the temporal dimension, followed by randomly splicing them back [1]_.
    By default, augmentation is performed on `edata` and `label`. Ensure the
    availability of the data.

    Parameters
    ----------
    samples : int
        The number of consecutive samples to segment the data. eg, 125 for
        250Hz data is segmented by 0.5s.
    multiply : float
        Data expansion multiple of relative metadata, 1 means doubled.
    only_train : bool
        If True, data augmentation is performed only on the training set.
    strict : bool
        If False, allow the input data to be unsplit. In this case, data
        augmentation will be applied to all data.
    shuffle : bool
        Whether or not to shuffle the data after picking.
    seed : int
        Controls the shuffling applied to the data after picking.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    References
    ----------

    .. [1] F. Lotte, “Signal processing approaches to minimize or suppress
        calibration time in oscillatory activity-based brain–computer
        interfaces,” Proceedings of the IEEE, vol. 103, no. 6,
        pp. 871–890, 2015.

    Notes
    -----
    Data augmentation is only applied to the `edata` and `label` within the
    eegdata, with other values remaining unchanged. If there are derived values
    based on the `edata`, attention should be paid to the order of
    transformations.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> split_eegdata = dpeeg.SplitEEGData(eegdata.copy(), eegdata.copy())
    >>> transforms.SegRecTime(2, 3)(split_eegdata, verbose=False)
    Train: [edata=(64, 3, 10), label=(64,)]
    Test : [edata=(16, 3, 10), label=(16,)]
    """

    def __init__(
        self,
        samples: int,
        multiply: float = 1.0,
        only_train: bool = True,
        strict: bool = True,
        shuffle: bool = True,
        seed: int = DPEEG_SEED,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"), only_train, strict)
        self.samples = samples
        self.multiply = multiply
        self.shuffle = shuffle
        self.seed = seed

    def _apply_aug(self, eegdata: EEGData, mode: str, verbose=None) -> EEGData:
        eegdata["edata"], eegdata["label"] = segmentation_and_reconstruction_time(
            data=eegdata["edata"],
            label=eegdata["label"],
            samples=self.samples,
            multiply=self.multiply,
            shuffle=self.shuffle,
            seed=self.seed,
            verbose=verbose,
        )

        return eegdata


class SlideWinAug(Augmentation):
    """Sliding window data augmentation.

    Data augmentation based on sliding windows will apply sliding windows to
    the training set and crop the corresponding time windows in the test set.
    By default, augmentation is performed on `edata` and `label`. Ensure the
    availability of the data.

    Parameters
    ----------
    win : int
        The size of the sliding window.
    overlap : int
        The amount of overlap between adjacent sliding windows.
    tmin : int
        Start time of selection in sampling points.
    tmax : int, optional
        End time of selection in sampling points. The default is to use the
        window length from the start time.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    Notes
    -----
    Data augmentation is only applied to the `edata` and `label` within the
    eegdata, with other values remaining unchanged. If there are derived values
    based on the `edata`, attention should be paid to the order of
    transformations.

    Examples
    --------
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> split_eegdata = dpeeg.SplitEEGData(eegdata.copy(), eegdata.copy())
    >>> transforms.SlideWinAug(2)(split_eegdata, verbose=False)
    Train: [edata=(80, 3, 2), label=(80,)]
    Test : [edata=(16, 3, 2), label=(16,)]
    """

    def __init__(
        self,
        win: int,
        overlap: int = 0,
        tmin: int = 0,
        tmax: int | None = None,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"), False, True)
        self.win = win
        self.overlap = overlap
        self.tmin = tmin
        self.tmax = tmin + win if tmax is None else tmax

    def _apply_aug(self, eegdata: EEGData, mode: str, verbose=None) -> EEGData:
        if mode == "train":
            eegdata["edata"], eegdata["label"] = slide_win(
                data=eegdata["edata"],
                win=self.win,
                overlap=self.overlap,
                label=eegdata["label"],
                verbose=verbose,
            )
        else:
            eegdata["edata"] = crop(
                data=eegdata["edata"],
                tmin=self.tmin,
                tmax=self.tmax,
                include_tmax=False,
                verbose=verbose,
            )

        return eegdata
