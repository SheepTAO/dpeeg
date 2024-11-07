# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod
from typing import overload, Literal

from mne.utils import verbose, logger
import test

from ..datasets.base import (
    _DataAlias,
    _DataVar,
    BaseData,
    BaseDataset,
    EEGDataset,
    EEGData,
    MultiSessEEGData,
    SplitEEGData,
)
from ..utils import unpacked, DPEEG_SEED, get_init_args, iterable_to_str


class Transforms(ABC):
    """Base class for eegdata transformation.

    The base class is callable, and the `__call__` function internally calls
    the `_apply` function according to the input data type. The allowed input
    types are eegdata or eegdataset. It also supports returning the transformed
    data of all subjects in the dataset and iteratively returning the
    transformed data of all subjects in the dataset and iteratively returning
    the transformed data of each subject. Transformation methods that inherit
    from this base class only need to override the `_apply` function.

    Attributes
    ----------
    subject : int, None
        When the input type is a dataset, if refers to the subject currently
        being operated on. Otherwise, it is None.
    """

    def __init__(self, repr: str | None = None) -> None:
        self._repr = repr

    @overload
    def __call__(self, eegdata: BaseData, verbose=None):
        """Apply data transformation to eegdata."""
        pass

    @overload
    def __call__(
        self, eegdata: BaseDataset, iter: Literal[False] = False, verbose=None
    ):
        """Apply data transformation to eegdataset."""
        pass

    @overload
    def __call__(self, eegdata: BaseDataset, iter: Literal[True] = True, verbose=None):
        """Apply data transformation to eegdataset."""
        pass

    @verbose
    def __call__(self, eegdata, iter: bool = False, verbose=None):  # type: ignore
        """Apply data transformation to eegdata or eegdataset.

        Parameters
        ----------
        eegdata : :ref:`eeg_data`, :ref:`eeg_dataset`
            Apply data transformation to eegdata or eegdataset.
        iter : bool
            Valid when the input type is eegdataset. ``False`` means directly
            returning the entire transformed dataset, ``True`` means
            iteratively returning the transformed data of each subject.

        Examples
        --------
        Allows transformation of data of type eegdata:

        >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
        ...                         label=np.random.randint(0, 3, 16))
        >>> transforms.Unsqueeze()(eegdata, verbose=False)
        [edata=(16, 1, 3, 10), label=(16,)]

        Also allows input type eegdataset:

        >>> eegdataset = dpeeg.datasets.EEGDataset([
        ...     eegdata.copy(), eegdata.copy(), eegdata.copy()
        ... ])
        >>> transforms.Squeeze()(eegdataset, iter=False, verbose=False)
        [EEGDataset:
          [eegdataset]: Subjects=3, type=EEGData
          [event_id]: None
        ]

        setting `iter` can iteratively obtain the eegdata of each subject after
        transformation:

        >>> tran = transforms.Unsqueeze()
        >>> for subject, eegdata in tran(eegdataset, iter=True, verbose=False):
        ...     print(subject, eegdata)
        0 [edata=(16, 1, 3, 10), label=(16,)]
        1 [edata=(16, 1, 3, 10), label=(16,)]
        2 [edata=(16, 1, 3, 10), label=(16,)]
        """
        if isinstance(eegdata, BaseData):
            return self._apply_wrap(eegdata, None)

        elif isinstance(eegdata, BaseDataset):
            if iter:
                return (
                    (subject, self._apply_wrap(egd, subject))
                    for subject, egd in eegdata.items()
                )
            else:
                dataset = EEGDataset(
                    event_id=eegdata.event_id, rename=eegdata._repr["_obj_name"]
                )
                for subject, egd in eegdata.items():
                    logger.info(f"Transform subject {subject}")
                    dataset[subject] = self._apply_wrap(egd, subject)
                return dataset

        else:
            raise TypeError("Unsupported eegdata type.")

    def _apply_wrap(self, eegdata: BaseData, subject: int | None) -> BaseData:
        logger.info(f"Apply {self} ...")
        self.subject = subject
        return self._apply(eegdata)

    @abstractmethod
    def _apply(self, eegdata: BaseData) -> BaseData:
        pass

    def __repr__(self) -> str:
        if self._repr:
            return self._repr
        else:
            class_name = self.__class__.__name__
            return f"{class_name} not implement attribute `self._repr`."


class TransformsEGD(Transforms):
    """This is a simple base class inherited from ``Transforms``, which only
    overrides the ``_apply`` method. If the transformation is indifferent to
    data types, one can inherit from this class and override the ``_apply_egd``
    method.
    """

    def _apply(self, eegdata: BaseData) -> BaseData:
        for egd, key in eegdata._datas():
            self._apply_egd(egd, key)
        return eegdata

    @abstractmethod
    def _apply_egd(self, egd: EEGData, key: str | None):
        pass


class Sequential(Transforms):
    """A sequential container.

    Transforms will be added to it in the order they are passed.

    Parameters
    ----------
    transforms : sequential of :doc:`../api/transforms`
        Sequential of transforms to compose.

    Examples
    --------
    If you have multiple transforms that are processed sequentiallt, you can do
    like:

    >>> transforms.Sequential(
    ...     transforms.Unsqueeze(),
    ...     transforms.Crop(2, 5),
    ... )
    >>> trans
    Sequential(
     (0): Unsqueeze(key=edata, dim=1)
     (1): Crop(tmin=2, tmax=5, include_tmax=False)
    )

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> trans(eegdata, verbose=False)
    [edata=(16, 1, 3, 3), label=(16,)]
    """

    def __init__(self, *transforms: Transforms) -> None:
        super().__init__()
        self.trans: list[Transforms] = []
        self.appends(*transforms)

    def __getitem__(self, key):
        """Return the transform of key."""
        return self.trans[key]

    def _apply_wrap(self, eegdata: BaseData, subject: int | None) -> BaseData:
        self.subject = subject
        return self._apply(eegdata)

    def _apply(self, eegdata: BaseData) -> BaseData:
        for tran in self.trans:
            eegdata = tran._apply_wrap(eegdata, self.subject)
        return eegdata

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
    :class:`.SplitEEGData`, no processing is done. For inputs that are
    :class:`.MultiSessEEGData`, data can be split by session or merged and then
    split. For :class:`.EEGData` inputs, data is split proportionally.

    Parameters
    ----------
    test_size : float
        The proportion of the test set. Default use stratified fashion.
    cross : bool
        ``True`` indicates that data from multiple sessions will be split into
        training and test sets, working in conjunction with ``train_sessions``
        and ``test_sessions``. ``False`` indicates that data from multiple
        sessions will be merged and then split into training and test sets,
        working with ``train_sessions``. These parameter are only effective
        when the input data type is ``MultiSessEEGData``; they are ignored for
        other types.
    train_sessions : list of str, optional
        Session data to be used as the training set.
        If ``cross=False``, ``train_sessions`` represents the sessions to be
        mixed and split (If ``None``, all session data will be used.).
        If ``cross=True``, ``train_sessions`` must be specified and represents
        the sessions to be used as the training set.
    test_sessions : list of str, optional
        Session data to be used as the test set.
        If ``cross=False``, this parameter is ignored.
        If ``cross=True``, ``test_sessions`` represents the sessions to be used
        as the test set (If ``None``, the complement of `train_sessions` will
        be used as the test set.).
    keys : list of str, optional
        The key of the eegdata value to be split. If ``None``, all data will be
        split, and it is necessary to ensure the uniformity of the data
        samples. Ignored when ``cross=True``.
    seed : int
        Random seed when splitting.

    Returns
    -------
    split_eegdata : :class:`.SplitEEGData`
        Split eegdata or dataset.

    Examples
    --------
    Split the eegdata:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> transforms.SplitTrainTest(0.2)(eegdata, verbose=False)
    Train: [edata=(12, 3, 10), label=(12,)]
    Test : [edata=(4, 3, 10), label=(4,)]

    Split eegdata across multiple sessions:

    >>> multi_sess_eegdata = dpeeg.MultiSessEEGData(
    ...     [eegdata.copy() for _ in range(4)])
    >>> transforms.SplitTrainTest(
    ...     cross=True,
    ...     train_sessions=[f"session_{i + 1}" for i in range(2)],
    ...     test_sessions=[f"session_4"]
    ... )(multi_sess_eegdata, verbose=False)
    Train: [edata=(32, 3, 10), label=(32,)]
    Test : [edata=(16, 3, 10), label=(16,)]

    or split the merged multiple sessions eegdata:

    >>> transforms.SplitTrainTest(0.5)(multi_sess_eegdata, verbose=False)
    Train: [edata=(32, 3, 10), label=(32,)]
    Test : [edata=(32, 3, 10), label=(32,)]
    """

    def __init__(
        self,
        test_size: float = 0.25,
        cross: bool = False,
        train_sessions: list[str] | None = None,
        test_sessions: list[str] | None = None,
        keys: list[str] | None = None,
        seed: int = DPEEG_SEED,
    ) -> None:
        super().__init__(get_init_args(self, locals(), format="rp"))
        assert 0 < test_size < 1, "`test_size` should be between 0 and 1."

        self.test_size = test_size
        self.cross = cross
        self.seed = seed
        self.keys = keys
        self.train_sessions = None if train_sessions is None else set(train_sessions)
        self.test_sessions = None if test_sessions is None else set(test_sessions)

    def _apply(self, eegdata: BaseData) -> SplitEEGData:
        if isinstance(eegdata, SplitEEGData):
            return eegdata
        elif isinstance(eegdata, MultiSessEEGData):
            return self._apply_multi_sess_eegdata(eegdata)
        elif isinstance(eegdata, EEGData):
            return self._apply_eegdata(eegdata)
        else:
            raise ValueError(
                "Input should be `SplitEEGData`, `MultiSessEEGData` or `EEGData`, "
                f"but got {type(eegdata)}."
            )

    def _apply_eegdata(self, eegdata: EEGData) -> SplitEEGData:
        from sklearn.model_selection import train_test_split

        label = eegdata["label"]
        keys, values = [], []
        for key, value in eegdata.items():
            if (self.keys is None) or (key in self.keys):
                keys.append(key)
                values.append(value)

        splitting = train_test_split(
            *values, test_size=self.test_size, random_state=self.seed, stratify=label
        )

        i = 0
        train_egd, test_egd = EEGData(), EEGData()
        for key in keys:
            train_egd[key], test_egd[key] = splitting[i], splitting[i + 1]
            i += 2

        return SplitEEGData(train_egd, test_egd)

    def _merge_multi_sess_eegdata(
        self, eegdata: MultiSessEEGData, sessions: set
    ) -> EEGData:
        egd = EEGData()
        for session in sessions:
            egd.append(eegdata[session])
        return egd

    def _apply_multi_sess_eegdata(self, eegdata: MultiSessEEGData) -> SplitEEGData:
        sessions = set(eegdata.keys())

        if self.cross:
            # check `self.train_sessions`
            if self.train_sessions is None:
                raise ValueError(
                    "`train_sessions` is needed when the eegdata is "
                    "`MultiSessEEGData` and `cross=True`."
                )

            train_inter = self.train_sessions & sessions
            if len(train_inter) == 0:
                raise KeyError(
                    f"Cannot find {iterable_to_str(self.train_sessions)}, "
                    f"eegdata only contains {iterable_to_str(sessions)}."
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

            train_eegdata = self._merge_multi_sess_eegdata(eegdata, train_inter)
            test_eegdata = self._merge_multi_sess_eegdata(eegdata, test_sessions)
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

            data = SplitEEGData(EEGData(), EEGData())
            for session in train_sessions:
                split = self._apply_eegdata(eegdata[session])
                data["train"].append(split["train"])
                data["test"].append(split["test"])

        return data


class ToEEGData(Transforms):
    """Convert different types of eegdata to :class:`.EEGData`.

    Merge all internal EEGData data of SplitEEGData and MultiSessEEGData
    together. This is done by calling :meth:`.EEGData.append`.

    Examples
    --------
    Convert :class:`.SplitEEGData` to :class:`.EEGData`:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                             label=np.random.randint(0, 3, 16))
    >>> split_eegdata = dpeeg.SplitEEGData(eegdata, eegdata)
    >>> transforms.ToEEGData()(split_eegdata, verbose=False)
    [edata=(32, 3, 10), label=(32,)]

    Convert :class:`.MultiSessEEGData` to :class:`.EEGData`:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> multi_sess_eegdata = dpeeg.MultiSessEEGData([eegdata, eegdata])
    >>> transforms.ToEEGData()(multi_sess_eegdata, verbose=False)
    [edata=(32, 3, 10), label=(32,)]
    """

    def __init__(self) -> None:
        super().__init__("ToEEGData()")

    def _apply(self, eegdata: BaseData) -> EEGData:
        if isinstance(eegdata, EEGData):
            return eegdata
        elif isinstance(eegdata, MultiSessEEGData):
            logger.info(f" `MultiSessEEGData` to `EEGData`")
            return self._apply_multi_sess_eegdata(eegdata)
        elif isinstance(eegdata, SplitEEGData):
            logger.info(f" `SplitEEGData` to `EEGData`")
            return self._apply_split_eegdata(eegdata)
        else:
            raise ValueError(
                "Input should be `SplitEEGData`, `MultiSessEEGData` or `EEGData`, "
                f"but got {type(eegdata)}."
            )

    def _apply_split_eegdata(self, eegdata: SplitEEGData) -> EEGData:
        egd = EEGData()
        egd.append(eegdata["train"])
        egd.append(eegdata["test"])
        return egd

    def _apply_multi_sess_eegdata(self, eegdata: MultiSessEEGData) -> EEGData:
        egd = EEGData()
        for tmp_egd, _ in eegdata._datas():
            egd.append(tmp_egd)
        return egd


def merge_subjects(
    eegdata: BaseDataset,
    subjects: list[int] | set[int],
    prefix: str = "",
    ret_eegdata: bool = True,
):
    if len(prefix) != 0:
        prefix = f"{prefix}_"

    multi_sess_eegdata = MultiSessEEGData({})
    for subject in subjects:
        egd = eegdata[subject]
        ps = f"{prefix}{subject}"

        if isinstance(egd, EEGData):
            multi_sess_eegdata[ps] = egd
        elif isinstance(egd, MultiSessEEGData):
            for k, v in egd.items():
                multi_sess_eegdata[f"{ps}_{k}"] = v
        elif isinstance(egd, SplitEEGData):
            multi_sess_eegdata[f"{ps}_train"] = egd["train"]
            multi_sess_eegdata[f"{ps}_test"] = egd["test"]

    if ret_eegdata:
        return ToEEGData()._apply_multi_sess_eegdata(multi_sess_eegdata)
    else:
        return multi_sess_eegdata


@overload
def split_subjects(
    eegdata: BaseDataset,
    test_subjects: list[int] | None = None,
    ret_eegdata: Literal[True] = True,
    verbose=None,
) -> EEGData | SplitEEGData:
    pass


@overload
def split_subjects(
    eegdata: BaseDataset,
    test_subjects: list[int] | None = None,
    ret_eegdata: Literal[False] = False,
    verbose=None,
) -> MultiSessEEGData:
    pass


@verbose
def split_subjects(
    eegdata: BaseDataset,
    test_subjects: list[int] | None = None,
    ret_eegdata: bool = True,
    verbose=None,
):
    """Split the dataset by subjects.

    Splitting the dataset at the subject level is different from the
    :class:`.SplitTrainTest` transformation. The former splits the data of all
    subjects in the entire dataset (similar to cross-subject), while the latter
    splits the data of each subject. The eegdata of different subjects are
    converted through :class:`ToEEGData` when ``ret_eegdata=True``.

    Parameters
    ----------
    eegdata : :ref:`eeg_dataset`
        Input eeg dataset.
    test_subjects : list of int, optional
        The list of subjects in the test set. If ``None``, the subject data of
        the entire dataset will be merged.
    ret_eegdata : bool
        Transform the merged inter-subject data into :class:`.EEGData` type. If
        ``False``, return labeled MultiSessEEGData. For specific usage, please
        refer to the example.


    .. attention::
        Since this transformation will change the structure of the entire
        dataset, it cannot be used with :class:`Sequential`. It is often used
        at the begining or end of preprocessing the dataset.

    Examples
    --------
    Since the input is a dataset type, first define a dataset with 2 subjects.
    Here, merge the data of all subjects:

    >>> from dpeeg.datasets import EEGDataset
    >>>
    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> multi_sess_eegdata = dpeeg.MultiSessEEGData([eegdata, eegdata])
    >>> eegdataset = EEGDataset([eegdata, eegdata, multi_sess_eegdata])
    >>> transforms.split_subjects(eegdataset, verbose=False)
    [edata=(64, 3, 10), label=(64,)]

    set ``train_subjects`` parameters to split the dataset:

    >>> transforms.split_subjects(eegdataset, [1, 2], verbose=False)
    Train: [edata=(32, 3, 10), label=(32,)]
    Test : [edata=(32, 3, 10), label=(32,)]

    set ``ret_eegdata`` to return the labeled MultiSessEEGData:

    >>> transforms.split_subjects(eegdataset, [3], False, verbose=False)
    {'train_1': [edata=(16, 3, 10), label=(16,)],
     'train_2': [edata=(16, 3, 10), label=(16,)],
     'test_3_session_1': [edata=(16, 3, 10), label=(16,)],
     'test_3_session_2': [edata=(16, 3, 10), label=(16,)]}
    """
    if not isinstance(eegdata, BaseDataset):
        raise TypeError(f"Input must be dataset.")

    if test_subjects is None:
        logger.info("Merge all subject eegdata.")
        return merge_subjects(eegdata, eegdata.keys(), ret_eegdata=ret_eegdata)
    else:
        subjects = set(eegdata.keys())
        te_subjects = set(test_subjects)
        inter = subjects & te_subjects
        compl = subjects - inter

        if len(inter) == 0:
            raise ValueError(
                f"Cannot find {iterable_to_str(te_subjects)} subjects, "
                f"the dataset contains {iterable_to_str(compl)} subjects."
            )

        logger.info("Split the subject eegdata.")
        if ret_eegdata:
            tr_egd = merge_subjects(eegdata, compl, ret_eegdata=True)
            te_egd = merge_subjects(eegdata, inter, ret_eegdata=True)
            ret_egd = SplitEEGData(tr_egd, te_egd)  # type: ignore
        else:
            ret_egd = MultiSessEEGData([])
            ret_egd.update(merge_subjects(eegdata, compl, "train", False))
            ret_egd.update(merge_subjects(eegdata, inter, "test", False))

        return ret_egd
