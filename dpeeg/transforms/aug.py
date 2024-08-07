# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import abstractmethod

from mne.utils import verbose, logger
from torch import mode

from ..datasets.base import _BaseData, SplitEEGData
from .base import Transforms
from ..utils import get_init_args
from .functions import segmentation_and_reconstruction_time, slide_win, crop


class Augmentation(Transforms):
    """Data augmentation base class.

    Augment the data, with default augmentation applied to the `edata` and
    `label`. Please verify the validity of the data.

    Parameters
    ----------
    only_train : bool
        If True, data augmentation is performed only on the training set.
    order : bool
        If False, allow the input data to be unsplit. In this case, data
        augmentation will be applied to all data.
    """

    def __init__(
        self, repr: str | None = None, only_train: bool = True, order: bool = True
    ) -> None:
        super().__init__(repr)
        self.only_train = only_train
        self.order = order

    def _apply(self, input: _BaseData, verbose=None) -> _BaseData:
        if (not isinstance(input, SplitEEGData)) or (not self.order):
            raise TypeError("The input must have been split, or the `order` is closed.")
        return self._apply_aug(input, verbose)

    @abstractmethod
    def _apply_aug(self, input: _BaseData, verbose=None) -> _BaseData:
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
    order : bool
        If False, allow the input data to be unsplit. In this case, data
        augmentation will be applied to all data.

    References
    ----------

    .. [1] F. Lotte, “Signal processing approaches to minimize or suppress
            calibration time in oscillatory activity-based brain–computer
            interfaces,” Proceedings of the IEEE, vol. 103, no. 6,
            pp. 871–890, 2015.
    """

    def __init__(
        self,
        samples: int,
        multiply: float = 1.0,
        only_train: bool = True,
        order: bool = True,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"), only_train, order)
        self.samples = samples
        self.multiply = multiply

    def _apply_aug(self, input: _BaseData, verbose=None) -> _BaseData:
        for eegdata, mode in input.datas():
            if (mode == "train") and self.only_train:
                continue

            eegdata["edata"], eegdata["label"] = segmentation_and_reconstruction_time(
                eegdata["edata"], eegdata["label"], self.samples, self.multiply, verbose
            )
        return input


class SlideWinAug(Augmentation):
    """Sliding window data augmentation.

    Data augmentation based on sliding windows will apply sliding windows to
    the training set and crop the corresponding time windows in the test set.

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
    """

    def __init__(
        self,
        win: int,
        overlap: int = 0,
        tmin: int = 0,
        tmax: int | None = None,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"), True, True)
        self.win = win
        self.overlap = overlap
        self.tmin = tmin
        self.tmax = tmin + win if tmax is None else tmax

    def _apply_aug(self, input: _BaseData, verbose=None) -> _BaseData:
        for eegdata, mode in input.datas():
            if mode == "train":
                eegdata["edata"], eegdata["label"] = slide_win(
                    eegdata["edata"], self.win, self.overlap, eegdata["label"], verbose
                )
            else:
                eegdata["edata"] = crop(
                    eegdata["edata"], self.tmin, self.tmax, verbose=verbose
                )

        return input
