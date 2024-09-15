# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Any, TypeAlias, TypeVar, overload
from copy import deepcopy, copy

import numpy as np
from numpy import ndarray

from ..utils import get_init_args, iterable_to_str, mapping_to_str, _format_log


class BaseData(ABC):
    """The bottom-level base data class."""

    @abstractmethod
    def _datas(self) -> Generator[tuple["EEGData", Any], None, None]:
        pass

    def copy(self):
        """Creates a shallow copy of the current object."""
        return copy(self)


_BaseData = TypeVar("_BaseData", bound=BaseData)


class EEGData(BaseData):
    """The base eegdata class.

    The most basic eeg data wrapper. It is essentially a python ``dict``, but
    with some additional functions added to it.

    Parameters
    ----------
    edata : ndarray, optional
        EEG data.
    label : ndarray, optional
        The labels corresponding to the eeg data.
    strict : bool
        ``True`` means that the number of ``edata`` and ``label`` samples must
        be the same.
    **kwargs : dict, optional
        Other parameter indicators.

    Examples
    --------
    If no value is passed, an empty EEGData object is initialized:

    >>> dpeeg.EEGData()
    [edata=(0,), label=(0,)]

    Or initialize with additional parameters:

    >>> dpeeg.EEGData(
    ...     edata=np.random.rand(16, 3, 20),
    ...     label=np.random.randint(0, 3, 16),
    ...     adj=np.random.rand(3, 3)
    ... )
    [edata=(16, 3, 20), label=(16,), adj=(3, 3)]
    """

    def __init__(
        self,
        edata: ndarray | None = None,
        label: ndarray | None = None,
        strict: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        edata = np.empty(0) if edata is None else edata
        label = np.empty(0) if label is None else label
        if strict:
            self._check(edata, label)

        self.data = {"edata": edata, "label": label, **kwargs}

    def _check(self, edata, label, raise_error=True):
        if len(edata) != len(label):
            if raise_error:
                raise ValueError(
                    f"The length {len(edata)} of edata is not equal to "
                    f"the length {len(label)} of label."
                )
            else:
                return False

        return True

    def __getitem__(self, key) -> ndarray:
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        """D.keys() -> a set-like object providing a view on D's keys"""
        return self.data.keys()

    def values(self):
        """D.values() -> an object providing a view on D's values"""
        return self.data.values()

    def items(self):
        """D.items() -> a set-like object providing a view on D's items"""
        return self.data.items()

    def pop(self, key, default, /):
        """D.pop(k[,d]) -> v, remove specified key and return the value

        If the key is not found, return the default if given; otherwise,
        raise a KeyError.
        """
        if key in ["edata", "label"]:
            raise KeyError(f"{key} as a basic structure cannot be popped")
        self.data.pop(key, default)

    def __len__(self):
        """Returns the number of internal keys, not the number of samples."""
        return len(self.data)

    def check(self) -> bool:
        """Check whether the number of samples of the current `edata` and
        `label` is equal.
        """
        return self._check(self.data["edata"], self.data["label"], False)

    def trials(self) -> int:
        """Returns the number of trial samples."""
        return len(self.data["label"])

    def index(self, idx: Iterable[int] | slice) -> "EEGData":
        """Index internal data and return a new EEGData instance.

        Parameters
        ----------
        idx : Iterable[int], slice
            The index to apply to the internal data.

        Returns
        -------
        EEGData
            A new EEGData instance containing the indexed data.

        Examples
        --------
        >>> eegdata = dpeeg.EEGData(
        ...     edata=np.random.randn(16, 3, 10),
        ...     label=np.random.randint(0, 3, 16),
        ... )
        >>> eegdata.index([1, 2, 3])
        [edata=(3, 3, 10), label=(3,)]

        >>> eegdata.index(slice(6))
        [edata=(6, 3, 10), label=(6,)]
        """
        eegdata = EEGData()
        for key, value in self.data.items():
            eegdata[key] = value[idx]
        return eegdata

    def _datas(self) -> Generator[tuple["EEGData", None], None, None]:
        yield self, None

    def append(
        self, eegdata: "EEGData", dims: list[int] | int = 0, ignore: bool = False
    ):
        """Append a new eegdata at the end of the eegdata.

        Parameters
        ----------
        eegdata : EEGData
            The new eegdata to be added. If the new eegdata contains key-value
            pairs that do not exist in the original eegdata, they will be
            ignored by default.
        dims : int, list of int
            The dimension along which the new eegdata is merged with the old
            eegdata. If a single number is entered, all eegdata is added along
            that dimension. If a list is given, the new eegdata is added along
            the specified dimensions in the order given in the list and
            dictionary.
        ignore : bool
            If `True`, add new key-value pairs from the new eegdata.

        Examples
        --------
        >>> eegdata = dpeeg.EEGData(
        ...     edata=np.random.randn(16, 3, 10),
        ...     label=np.random.randint(0, 3, 16),
        ... )
        >>> eegdata.append(eegdata.copy())
        >>> eegdata
        [edata=(32, 3, 10), label=(32,)]

        >>> eegdata_adj = eegdata.copy()
        >>> eegdata_adj["adj"] = np.random.randn(16, 3, 3)
        >>> eegdata_adj
        [edata=(32, 3, 10), label=(32,), adj=(16, 3, 3)]

        >>> eegdata.append(eegdata_adj, ignore=True)
        >>> eegdata
        [edata=(64, 3, 10), label=(64,), adj=(16, 3, 3)]
        """
        if isinstance(dims, int):
            dims = [dims] * len(eegdata)

        keys = self.keys()
        for i, key in enumerate(eegdata.keys()):
            if (key in keys) or (not ignore):
                self.add(key, eegdata[key], dim=dims[i])
            else:
                self[key] = eegdata[key]

    def add(self, key: str, value: ndarray | list[ndarray], dim: int = 0):
        """Add a new value to the corresponding key.

        Parameters
        ----------
        key : str
            The key of eegdata.
        value : ndarray, list of ndarray
            The new value to be added. If a `list`, all values in the list are
            added.
        dim : int
            The dimension in which the new data is concatenated with the old
            data.

        Examples
        --------
        >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
        ...                         label=np.random.randint(0, 3, 16))
        >>> eegdata.add("edata", np.random.randn(16, 3, 10), dim=2)
        >>> eegdata
        [edata=(16, 3, 20), label=(16,)]
        """
        if len(self[key]) == 0:
            if isinstance(value, list):
                self[key] = np.concatenate(value, axis=dim)
            else:
                self[key] = value

        else:
            data = [self[key]]
            if isinstance(value, ndarray):
                data.append(value)
            else:
                data.extend(value)
            self[key] = np.concatenate(data, axis=dim)

    @property
    def ncls(self) -> int:
        """Returns the number of categories."""
        return len(np.unique(self["label"]))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the ``edata`` shape."""
        return self["edata"].shape[1:]

    def __repr__(self) -> str:
        data = {key: value.shape for key, value in self.items()}
        return f"[{mapping_to_str(data)}]"


class MultiSessEEGData(BaseData, dict):
    """Multi-session eegdata.

    The multi-session EEGData is actually equivalent to the EEGData collection
    wrapped by the Python built-in ``dict``. It does not have to be multi-
    session data, it can be any other EEGData collection. This name is only
    used here to indicate this type of data. The actual format is similar to:

        multi_sess_eegdata = { 'session_id': EEGData }

    Parameters
    ----------
    data : list or dict of EEGData
        Initialize EEG data. If it is a list of ``EEGData``, add a key to each
        data in the order of the list. If it is a dict of ``EEGData``, use the
        dict to initialize.

    Examples
    --------
    Data type is list:

    >>> eegdata = dpeeg.EEGData(edata=np.random.randn(16, 3, 10),
    ...                         label=np.random.randint(0, 3, 16))
    >>> dpeeg.MultiSessEEGData([eegdata.copy(), eegdata.copy()])
    {'session_1': [edata=(16, 3, 10), label=(16,)],
     'session_2': [edata=(16, 3, 10), label=(16,)]}

    or the data type is custom dict:

    >>> dpeeg.MultiSessEEGData({'xx': eegdata.copy(), 'yy': eegdata.copy()})
    {'xx': [edata=(16, 3, 10), label=(16,)],
     'yy': [edata=(16, 3, 10), label=(16,)]}
    """

    @overload
    def __init__(self, data: list[EEGData]) -> None:
        pass

    @overload
    def __init__(self, data: dict[str, EEGData]) -> None:
        pass

    def __init__(self, data):
        if isinstance(data, list):
            data = {f"session_{i + 1}": session for i, session in enumerate(data)}
        elif isinstance(data, dict):
            pass
        else:
            raise TypeError(f"Input type {type(data)} cannot be parsed.")

        super().__init__(data)

    def __getitem__(self, key: str) -> EEGData:
        return super().__getitem__(key)

    def trials(self, sum: bool = True) -> list[int] | int:
        """Returns the sum of the number of trials for all data or the number
        of trials for each data.
        """
        trials = []
        for session in self.keys():
            trials.append(len(self[session]["label"]))
        if sum:
            return np.sum(trials)
        else:
            return trials

    def _datas(self) -> Generator[tuple[EEGData, str], None, None]:
        for session, eegdata in self.items():
            yield eegdata, session


class SplitEEGData(BaseData):
    """Split eegdata.

    The eegdata has been split into training and test sets. A dict of data is
    maintained internally.

    Parameters
    ----------
    train : EEGData
        Training set data.
    test : EEGData
        Test set data.
    strict : bool
        The training and test sets `edata` are required to have the same shape.
    """

    def __init__(self, train: EEGData, test: EEGData, strict: bool = True) -> None:
        super().__init__()

        if strict and (train.shape != test.shape):
            raise ValueError(
                f"The training data shape {train.shape} is not the same as "
                f"the test data shape {test.shape}."
            )

        self.data = {"train": train, "test": test}

    def __setitem__(self, key: str, value: EEGData) -> None:
        if key not in ["train", "test"]:
            raise KeyError("The key must be train or test")
        self.data[key] = value

    def __getitem__(self, key: str) -> EEGData:
        return self.data[key]

    def trials(self) -> tuple[int, int]:
        """Returns the number of trials for training and testing data
        respectively.
        """
        train_trials = len(self.data["train"]["label"])
        test_trials = len(self.data["test"]["label"])
        return train_trials, test_trials

    def _datas(self) -> Generator[tuple[EEGData, str], None, None]:
        for mode, eegdata in self.data.items():
            yield eegdata, mode

    def __repr__(self) -> str:
        return f"Train: {self.data['train']}\n" f"Test : {self.data['test']}"


_DataAlias: TypeAlias = EEGData | MultiSessEEGData | SplitEEGData
_DataVar = TypeVar("_DataVar", EEGData, MultiSessEEGData, SplitEEGData)


class BaseDataset(ABC):
    """The bottom-level base dataset class."""

    def __init__(
        self,
        repr: dict,
        event_id: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.event_id = event_id
        self._repr = repr

    @abstractmethod
    def keys(self) -> list[int]:
        pass

    @abstractmethod
    def _get_single_subject_data(self, subject: int) -> _DataVar:
        pass

    def __getitem__(self, subject: int):
        """Get the subject's eegdata."""
        return self._get_single_subject_data(subject)

    def __len__(self) -> int:
        return len(self.keys())

    def values(self):
        for subject in self.keys():
            yield self._get_single_subject_data(subject)

    def items(self):
        for subject in self.keys():
            subject_data = self._get_single_subject_data(subject)
            yield subject, subject_data

    def __repr__(self) -> str:
        return _format_log(self._repr)


_BaseDataset = TypeVar("_BaseDataset", bound=BaseDataset)


class EEGDataset(BaseDataset):
    """Base EEG dataset.

    Parameters
    ----------
    eegdataset : list or dict of eegdata, optional
        The eegdata of different subjects. If ``None``, initialize an empty
        dataset.
    event_id : dict, optional
        The correspondence between labels and events.
    rename : str, optional
        User renamed name. If ``None``, the default class name is used.

    Notes
    -----
    The dataset supports different subjects with different eegdata types (such
    as ``EEGData``, ``MultiSessEEGData`` and ``SplitEEGData``), but it is
    recommended to unify the eegdata types of all subjects when performing
    transformation and training models to avoid unpredictable errors.
    """

    @overload
    def __init__(
        self,
        eegdataset: list[BaseData] | None = None,
        event_id: dict[str, int] | None = None,
        rename: str | None = None,
    ) -> None:
        pass

    @overload
    def __init__(
        self,
        eegdataset: dict[int, BaseData] | None = None,
        event_id: dict[str, int] | None = None,
        rename: str | None = None,
    ) -> None:
        pass

    def __init__(self, eegdataset=None, event_id=None, rename=None) -> None:
        super().__init__(
            get_init_args(self, locals(), rename=rename, ret_dict=True),
            event_id=event_id,
        )
        self._repr.pop("rename")

        if eegdataset is None:
            self.eegdataset = {}
        elif isinstance(eegdataset, dict):
            self.eegdataset = eegdataset
        elif isinstance(eegdataset, list):
            self.eegdataset = {sub: egd for sub, egd in enumerate(eegdataset)}
        else:
            raise TypeError(f"Input type {type(eegdataset)} cannot be parsed.")

    def __setitem__(self, subject: int, eegdata: BaseData):
        """Set EEGData for subject.

        Parameters
        ----------
        subject : int, str
            Subject or session index.
        eegdata : EEGData

        """
        self.eegdataset[subject] = eegdata

    def _get_single_subject_data(self, subject: int):
        return self.eegdataset[subject]

    def get_data(self):
        """Returns the eegdata of all subjects."""
        data = {}
        for subject in self.keys():
            data[subject] = self._get_single_subject_data(subject)
        return data

    def keys(self):
        """Return the list of subjects."""
        return self.eegdataset.keys()

    def pop(self, subject: int, default=None):
        """Remove specified subject and return the corresponding eegdata.

        If the subject is not found, return the default if given; otherwise,
        raise a KeyError.
        """
        return self.eegdataset.pop(subject, default)

    def _eegdata_type(self) -> set[str]:
        type_set = set()
        for eegdata in self.eegdataset.values():
            type_set.add(type(eegdata).__name__)
        return type_set

    def __repr__(self) -> str:
        eegdataset = (
            f"Subjects={len(self.eegdataset)}"
            f", type={iterable_to_str(self._eegdata_type(), symbol='|')}"
        )
        self._repr["eegdataset"] = eegdataset
        return _format_log(self._repr)
