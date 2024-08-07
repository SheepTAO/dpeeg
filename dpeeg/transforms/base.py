# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod
from typing import overload

from mne.utils import verbose, logger

from ..datasets.base import (
    _DataAlias,
    BaseData,
    BaseDataset,
    EEGDataset,
    EEGData,
    MultiSessEEGData,
    SplitEEGData,
)
from ..utils import unpacked, DPEEG_SEED, get_init_args, iterable_to_str


class Transforms(ABC):
    def __init__(self, repr: str | None = None) -> None:
        self._repr = repr

    @overload
    def __call__(self, input: BaseData, verbose=None):
        pass

    @overload
    def __call__(self, input: BaseDataset, verbose=None):
        pass

    @verbose
    def __call__(self, input, verbose=None):
        if isinstance(input, BaseData):
            return self._apply(input, verbose)
        elif isinstance(input, BaseDataset):
            dataset = EEGDataset(
                event_id=input.event_id, rename=input.__class__.__name__
            )
            for subject, eegdata in input.items():
                dataset[subject] = self._apply(eegdata, verbose)
            return dataset
        else:
            raise TypeError("Unsupported input type.")

    @abstractmethod
    def _apply(self, input: BaseData, verbose=None) -> _DataAlias:
        pass

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            class_name = self.__class__.__name__
            return f"{class_name} not implement attribute `self._repr`."


class Sequential(Transforms):
    """A sequential container.

    Transforms will be added to it in the order they are passed.

    Parameters
    ----------
    transforms : sequential of `Transforms`
        Sequential of transforms to compose.

    Examples
    --------
    If you have multiple transforms that are processed sequentiallt, you can do
    like:
    >>> from dpeeg.data import transforms
    >>> trans = transforms.Sequential(
    ...     transforms.Unsqueeze(),
    ...     transforms.ToTensor(),
    ... )
    >>> trans
    Sequential(
        (0): Unsqueeze(dim=1)
        (1): ToTensor()
    )
    """

    def __init__(self, *transforms: Transforms) -> None:
        super().__init__()
        self.trans: list[Transforms] = []
        self.appends(*transforms)

    def _apply(self, input: _DataAlias, verbose=None) -> _DataAlias:
        for tran in self.trans:
            input = tran._apply(input, verbose)
        return input

    def __repr__(self) -> str:
        s = "Sequential("
        if len(self.trans) == 0:
            return s + ")"
        else:
            for idx, tran in enumerate(self.trans):
                s += f"\n ({idx}): {tran}"
        return s + "\n)"

    def appends(self, *transforms: Transforms) -> None:
        """Append transforms to the last of composes."""
        trans = unpacked(*transforms)
        for tran in trans:
            if isinstance(tran, Sequential):
                self.trans.extend(tran.get_transforms())
            else:
                self.trans.append(tran)

    def insert(self, index: int, transform: Transforms) -> None:
        """Insert a transform at index."""
        self.trans.insert(index, transform)

    def get_transforms(self) -> list[Transforms]:
        """Return list of Transforms."""
        return self.trans


class SplitTrainTest(Transforms):
    """Split the data into training and testing sets.

    Split different types of input data. For inputs that are already
    `SplitEEGData`, no processing is done. For inputs that are
    `MultiSessEEGData`, data can be split by session or merged and then split.
    For `EEGData` inputs, data is split proportionally.

    Parameters
    ----------
    cross : bool
        `True` indicates that data from multiple sessions will be split into
        training and test sets, working in conjunction with `train_sessions`
        and `test_sessions`. `False` indicates that data from multiple sessions
        will be merged and then split into training and test sets, working with
        `train_sessions`. These parameter are only effective when the input
        data type is `MultiSessEEGData`; they are ignored for other types.
    train_sessions : list of int, None
        Session data to be used as the training set.
        If `cross=False`, `train_sessions` represents the sessions to be mixed
        and split (If `None`, all session data will be used.).
        If `cross=True`, `train_sessions` must be specified and represents the
        sessions to be used as the training set.
    test_sessions : list of int, None
        Session data to be used as the test set.
        If `cross=False`, this parameter is ignored.
        If `cross=True`, `test_sessions` represents the sessions to be used as
        the test set (If `None`, the complement of `train_sessions` will be
        used as the test set.).
    test_size : float
        The proportion of the test set. Default use stratified fashion.
    seed : int
        Random seed when splitting.
    """

    def __init__(
        self,
        cross: bool = False,
        train_sessions: list[int] | None = None,
        test_sessions: list[int] | None = None,
        test_size: float = 0.25,
        seed: int = DPEEG_SEED,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))

        if train_sessions is not None:
            self.train_sessions = set([f"session_{i}" for i in train_sessions])
        else:
            self.train_sessions = None
        if test_sessions is not None:
            self.test_sessions = set([f"session_{i}" for i in test_sessions])
        else:
            self.test_sessions = None

        assert 0 < test_size < 1, "`test_size` should be between 0 and 1."
        self.cross = cross
        self.test_size = test_size
        self.seed = seed

    @verbose
    def _apply(self, input: BaseData, verbose=None) -> SplitEEGData:
        if isinstance(input, SplitEEGData):
            return input
        elif isinstance(input, MultiSessEEGData):
            return self._apply_multi_sess_eegdata(input, verbose)
        elif isinstance(input, EEGData):
            return self._apply_eegdata(input, verbose)
        else:
            raise ValueError(
                "Input should be `SplitEEGData`, `MultiSessEEGData` or `EEGData`, "
                f"but got {type(input)}."
            )

    @verbose
    def _apply_eegdata(self, input: EEGData, verbose=None) -> SplitEEGData:
        from sklearn.model_selection import train_test_split

        label = input["label"]
        train_eegdata, test_eegdata = EEGData(), EEGData()
        for key, value in input.items():
            train_eegdata[key], test_eegdata[key] = train_test_split(
                value, self.test_size, random_state=self.seed, stratify=label
            )

        return SplitEEGData(train_eegdata, test_eegdata)

    def _merge_multi_sess_eegdata(
        self, input: MultiSessEEGData, sessions: set
    ) -> EEGData:
        eegdata = EEGData()
        for session in sessions:
            eegdata.update(input[session])
        return eegdata

    @verbose
    def _apply_multi_sess_eegdata(
        self, input: MultiSessEEGData, verbose=None
    ) -> SplitEEGData:
        sessions = set(input.keys())

        if self.cross:
            # check `self.train_sessions`
            if self.train_sessions is None:
                raise ValueError(
                    "`train_sessions` is needed when the input is "
                    "`MultiSessEEGData` and `cross=True`."
                )

            train_inter = self.train_sessions & sessions
            if len(train_inter) == 0:
                raise KeyError(
                    f"Cannot find {iterable_to_str(self.train_sessions)}, "
                    f"input only contains {iterable_to_str(sessions)}."
                )
            train_compl = self.train_sessions - sessions
            if len(train_compl) != 0:
                logger.warning(
                    f"Cannot find {iterable_to_str(train_compl)}, "
                    f"only load {iterable_to_str(train_inter)} as train."
                )

            # check `self.test_sessions`
            if self.test_sessions is None:
                test_sessions = sessions - train_inter
            else:
                test_inter = self.test_sessions & sessions
                if len(test_inter) == 0:
                    test_sessions = sessions - train_inter
                    logger.warning(
                        f"Cannot find {iterable_to_str(self.test_sessions)}, "
                        f"only load {iterable_to_str(test_sessions)} as test."
                    )
                else:
                    test_sessions = test_inter
                    test_compl = self.test_sessions - sessions
                    if len(test_compl) != 0:
                        logger.warning(
                            f"Cannot find {iterable_to_str(test_compl)}, only "
                            f"load {iterable_to_str(test_sessions)} as test."
                        )

            train_eegdata = self._merge_multi_sess_eegdata(input, train_inter)
            test_eegdata = self._merge_multi_sess_eegdata(input, test_sessions)
            data = SplitEEGData(train_eegdata, test_eegdata)

        else:
            # check `self.train_sessions`
            if self.train_sessions is None:
                train_sessions = sessions
            else:
                train_inter = sessions & self.train_sessions
                if len(train_inter) == 0:
                    train_sessions = sessions
                    logger.warning(
                        f"Cannot find {iterable_to_str(self.train_sessions)}, "
                        f"use {iterable_to_str(train_sessions)} as mixed."
                    )
                else:
                    train_sessions = train_inter
                    train_compl = self.train_sessions - sessions
                    if len(train_compl) != 0:
                        logger.warning(
                            f"Cannot find {iterable_to_str(train_compl)}, only "
                            f"load {iterable_to_str(train_sessions)} as mixed."
                        )

            data = self._merge_multi_sess_eegdata(input, train_sessions)
            data = self._apply_eegdata(data, verbose)

        return data
