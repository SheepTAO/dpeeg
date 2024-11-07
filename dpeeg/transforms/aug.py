# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import abstractmethod

from mne.utils import verbose, logger

from ..datasets.base import _BaseData, EEGData, SplitEEGData
from .base import Transforms
from ..utils import DPEEG_SEED, get_init_args
from ..tools.docs import fill_doc
from .functions import (
    segmentation_and_reconstruction_time,
    slide_win,
    crop,
    gaussian_noise_time,
)


class Augmentation(Transforms):
    """Data augmentation base class.

    Augment the data, with default augmentation applied to the ``edata`` and
    ``label``. Please verify the validity of the data.
    """

    def __init__(
        self, repr: str | None = None, only_train: bool = True, strict: bool = True
    ) -> None:
        super().__init__(repr)
        self.only_train = only_train
        self.strict = strict

    def _apply(self, eegdata: _BaseData) -> _BaseData:
        if not isinstance(eegdata, SplitEEGData):
            if self.strict:
                raise TypeError(
                    "The input must have been split, or `strict` is set to False."
                )
            else:
                for egd, _ in eegdata._datas():
                    self._apply_aug(egd, "None")
        else:
            for egd, mode in eegdata._datas():
                if (mode != "train") and self.only_train:
                    continue
                self._apply_aug(egd, mode)

        return eegdata

    @abstractmethod
    def _apply_aug(self, egd: EEGData, mode: str):
        pass


@fill_doc
class SegRecTime(Augmentation):
    """Segmentation and reorganization in the time domain.

    The S&R process involves segmenting the original eeg signals based on class
    along the temporal dimension, followed by randomly splicing them back [1]_.
    By default, augmentation is performed on ``edata`` and ``label``. Ensure
    the availability of the data.

    Parameters
    ----------
    samples : int
        The number of consecutive samples to segment the data. eg, 125 for
        250Hz data is segmented by 0.5s.
    multiply : float
        Data expansion multiple of relative metadata, 1 means doubled.
    %(aug_only_train)s
    %(aug_strict)s
    seed : int
        Seed to be used to instantiate numpy random number generator instance.

    Returns
    -------
    data : eegdata or dataset
        Transformed eegdata.

    References
    ----------
    .. [1] F. Lotte, “Signal processing approaches to minimize or suppress
        calibration time in oscillatory activity-based brain-computer
        interfaces,” Proceedings of the IEEE, vol. 103, no. 6,
        pp. 871-890, 2015.

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
        seed: int = DPEEG_SEED,
    ) -> None:
        super().__init__(
            get_init_args(self, locals(), format="rp"),
            only_train=only_train,
            strict=strict,
        )
        self.samples = samples
        self.multiply = multiply
        self.seed = seed

    def _apply_aug(self, egd: EEGData, mode: str):
        egd["edata"], egd["label"] = segmentation_and_reconstruction_time(
            data=egd["edata"],
            label=egd["label"],
            samples=self.samples,
            multiply=self.multiply,
            seed=self.seed,
        )


class SlideWinAug(Augmentation):
    """Sliding window data augmentation.

    Data augmentation based on sliding windows will apply sliding windows to
    the training set and crop the corresponding time windows in the test set.
    By default, augmentation is performed on ``edata`` and ``label``. Ensure
    the availability of the data.

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
        super().__init__(
            get_init_args(self, locals(), format="rp"),
            only_train=False,
            strict=True,
        )
        self.win = win
        self.overlap = overlap
        self.tmin = tmin
        self.tmax = tmin + win if tmax is None else tmax

    def _apply_aug(self, egd: EEGData, mode: str):
        if mode == "train":
            egd["edata"], egd["label"] = slide_win(
                data=egd["edata"],
                win=self.win,
                overlap=self.overlap,
                label=egd["label"],
            )
        else:
            egd["edata"] = crop(
                data=egd["edata"],
                tmin=self.tmin,
                tmax=self.tmax,
                include_tmax=False,
            )


@fill_doc
class GaussTime(Augmentation):
    """Randomly add white noise to all channels.

    Gaussian white noise with a mean of 0 is directly added to the raw EEG
    signal as the generated new data [1]_. By default, augmentation is
    performed on ``edata`` and ``label``. Ensure the availability of the data.

    Parameters
    ----------
    std : float
        Standard deviation to use for the additive noise.
    %(aug_only_train)s
    %(aug_strict)s
    seed : int
        Seed to be used to instantiate numpy random number generator instance.

    References
    ----------
    .. [1] Wang, F., Zhong, S. H., Peng, J., Jiang, J., & Liu, Y. (2018). Data
       augmentation for eeg-based emotion recognition with deep convolutional
       neural networks. In International Conference on Multimedia Modeling
       (pp. 82-93).
    """

    def __init__(
        self,
        std: float,
        only_train: bool = True,
        strict: bool = True,
        seed: int = DPEEG_SEED,
    ) -> None:
        super().__init__(
            get_init_args(self, locals(), format="rp"),
            only_train=only_train,
            strict=strict,
        )
        self.std = std
        self.seed = seed

    def _apply_aug(self, egd: EEGData, mode: str):
        egd["edata"], egd["label"] = gaussian_noise_time(
            data=egd["edata"],
            label=egd["label"],
            mean=0,
            std=self.std,
            seed=self.seed,
        )
