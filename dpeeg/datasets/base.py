# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, TypeAlias, TypeVar, Generic, overload
from copy import deepcopy, copy

import numpy as np
from numpy import ndarray

from ..utils import get_init_args, mapping_to_str, _format_log


class BaseData(dict, ABC):

    @abstractmethod
    def datas(self) -> Generator[tuple["EEGData", Any], None, None]:
        pass

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)


_BaseData = TypeVar("_BaseData", bound=BaseData)


class EEGData(BaseData):
    """The bottom-level eegdata class.

    The most basic eeg data wrapper. It is essentially a python ``dict``, but
    with some additional functions added to it.

    Parameters
    ----------
    edata : ndarray
        EEG data.
    label : ndarray
        The labels corresponding to the eeg data.
    """

    def __init__(
        self, edata: ndarray | None = None, label: ndarray | None = None, **kwargs
    ) -> None:
        edata = np.empty(0) if edata is None else edata
        label = np.empty(0) if label is None else label
        super().__init__({"edata": edata, "label": label, **kwargs})

    def trials(self) -> int:
        """Returns the number of trial samples."""
        return len(self["label"])

    def datas(self) -> Generator[tuple["EEGData", Any], None, None]:
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
        >>> eegdata = dpeeg.EEGData(np.random.randn(16, 3, 10), np.random.randint(0, 3, 16))
        >>> eegdata.add("edata", np.random.randn(16, 3, 10), dim=2)
        >>> eegdata
        [edata=(16, 3, 20), label=(16,)]
        """
        data = [self[key]]
        if isinstance(value, ndarray):
            data.append(value)
        else:
            data.extend(value)
        self[key] = np.concatenate(data, axis=dim)

    def __repr__(self) -> str:
        data = {key: value.shape for key, value in self.items()}
        return f"[{mapping_to_str(data)}]"


class MultiSessEEGData(BaseData):
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
        trials = []
        for session in self.keys():
            trials.append(len(self[session]["label"]))
        if sum:
            return np.sum(trials)
        else:
            return trials

    def datas(self) -> Generator[tuple[EEGData, Any], None, None]:
        for session, eegdata in self.items():
            yield eegdata, session


class SplitEEGData(BaseData):
    def __init__(self, train: EEGData, test: EEGData) -> None:
        super().__init__({"train": train, "test": test})

    def __getitem__(self, key: str) -> EEGData:
        return super().__getitem__(key)

    def trials(self) -> tuple[int, int]:
        train_trials = len(self["train"]["label"])
        test_trials = len(self["test"]["label"])
        return train_trials, test_trials

    def datas(self) -> Generator[tuple[EEGData, Any], None, None]:
        for mode, eegdata in self.items():
            yield eegdata, mode


_DataAlias: TypeAlias = EEGData | MultiSessEEGData | SplitEEGData
_DataVar = TypeVar("_DataVar", EEGData, MultiSessEEGData, SplitEEGData)


class BaseDataset(ABC):
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
    def _get_single_subject_data(self, subject: int) -> _DataAlias:
        pass

    def __getitem__(self, subject: int):
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


class EEGDataset(BaseDataset, Generic[_BaseData]):
    """Base EEG dataset."""

    def __init__(
        self,
        eegdataset: dict[int, _BaseData] | None = None,
        event_id: dict[str, int] | None = None,
        rename: str | None = None,
    ) -> None:
        super().__init__(
            get_init_args(self, locals(), rename=rename, ret_dict=True),
            event_id=event_id,
        )
        self._repr.pop("rename")
        self.eegdataset = {} if eegdataset is None else eegdataset

    def __setitem__(self, subject: int, eegdata: _BaseData):
        """Set EEGData for subject.

        Parameters
        ----------
        subject : int, str
            Subject or session index.
        eegdata : EEGData

        """
        self.eegdataset[subject] = eegdata

    def _get_single_subject_data(self, subject: int) -> _BaseData:
        return self.eegdataset[subject]

    def get_data(self) -> dict[int, _BaseData]:
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

    def __repr__(self) -> str:
        self._repr["eegdataset"] = str(self.eegdataset)
        return _format_log(self._repr)
