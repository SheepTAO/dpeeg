# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from email.policy import strict
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Any, TypeAlias, TypeVar, overload
from copy import deepcopy

import mne
import numpy as np
from mne.io import Raw
from mne import Epochs
from tqdm import tqdm
from numpy import ndarray

from ..utils import (
    get_init_args,
    iterable_to_str,
    mapping_to_str,
    _format_log,
    DPEEG_DIR,
)


class BaseData(ABC):
    """The bottom-level base data class."""

    @abstractmethod
    def _datas(self) -> Generator[tuple["EEGData", Any], None, None]:
        pass

    def copy(self):
        """Creates a deep copy of the current object."""
        return deepcopy(self)


_BaseData = TypeVar("_BaseData", bound=BaseData)


class EEGData(BaseData):
    """The base eegdata class.

    The most basic eeg data wrapper. It is essentially a python ``dict``, but
    with some additional functions added to it.

    Parameters
    ----------
    edata : array shape (n_trials, ...), optional
        EEG data.
    label : array shape (n_trials, ...), optional
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
            If ``True``, add new key-value pairs from the new eegdata.

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
    def cls(self) -> list[int]:
        """Returns all categories."""
        return list(np.unique(self.data["label"]))

    @property
    def ncls(self) -> int:
        """Returns the number of categories."""
        return len(np.unique(self.data["label"]))

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the ``edata`` shape."""
        return self.data["edata"].shape[1:]

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

    .. Warning::

       Do not set the key value to the keywords "train" or "test"; otherwise,
       please use :class:`~SplitEEGData`.

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
        dataset. If ``list``, sort by subject one.
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
            self.eegdataset = {
                subject: egd for subject, egd in enumerate(eegdataset, start=1)
            }
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


DATA_PATH = Path(DPEEG_DIR) / "datasets"


class RawDataset(BaseDataset):
    """DPEEG Raw EEG Dataset.

    Datasets with ``mne.io.Raw`` as the base class, such as depression, use
    whether the subject suffers from depression as the label. The ``get_raw()``
    is used to obtain the raw EEG data of the subject, and ``get_data()`` is
    used to obtain data in the form of ``EEGData``. Since the raw EEGs of these
    datasets often have no labels, when calling ``get_data()``, you need to
    manually set the label through ``_set_label()`` according to different
    datasets (the label here is in units of run).
    """

    def __init__(
        self,
        repr: dict,
        subject_list: list[int],
        event_id: dict[str, int] | None = None,
        subjects: list[int] | None = None,
        tmin: float = 0.0,
        tmax: float | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        unit_factor: float = 1e6,
    ) -> None:
        super().__init__(repr, event_id)
        self.subjects = subject_list if subjects is None else subjects
        if tmax is not None and tmin >= tmax:
            raise ValueError("tmax must be greater than tmin")

        if not (set(self.subjects) <= set(subject_list)):
            raise KeyError(f"Subject must be between 1 and {subject_list[-1]}.")

        self.tmin = tmin
        self.tmax = tmax
        self.picks = picks
        self.resample = resample
        self.unit_factor = unit_factor

    def keys(self) -> list[int]:
        return self.subjects

    @abstractmethod
    def _get_single_subject_raw(self, subject: int, verbose="ERROR"):
        return NotImplementedError

    def get_raw(
        self, progressbar: bool = True, verbose="ERROR"
    ) -> dict[int, dict[str, dict[str, Raw]]]:
        """Return the raw correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id': {'session_id': {'run_id': Raw}}}

        subjects are on top, then we have sessions, then runs.
        A session is a recording done in a single day, without removing the EEG
        cap. A session is constitued of at least one run. A run is a single
        contigous recording. Some dataset break session in multiple runs.
        """
        subjects = tqdm(
            self.subjects,
            "Load Raw",
            unit="sub",
            dynamic_ncols=True,
            disable=not progressbar,
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_raw(subject, verbose)
        return data

    def _set_label(self, subject: int):
        return np.empty(0)

    def _get_single_subject_data(self, subject: int, verbose="ERROR"):
        data = []
        for session, runs in self._get_single_subject_raw(subject, verbose).items():  # type: ignore
            raws = []
            for run, raw in runs.items():
                if self.resample:
                    raw.resample(self.resample, verbose=verbose)

                if self.picks is None:
                    picks = mne.pick_types(raw.info, eeg=True, stim=False)
                else:
                    picks = mne.pick_channels(
                        raw.info["ch_names"], include=self.picks, ordered=True
                    )

                raws.append(
                    np.expand_dims(
                        raw.get_data(
                            picks=picks,
                            tmin=self.tmin,
                            tmax=self.tmax,
                            verbose=verbose,
                        ),
                        axis=0,
                    )
                )

            edata = np.concatenate(raws, axis=-1) * self.unit_factor
            label = self._set_label(subject)
            data.append(EEGData(edata, label))
        return MultiSessEEGData(data)

    def get_data(
        self, progressbar: bool = True, verbose="ERROR"
    ) -> dict[int, MultiSessEEGData]:
        """Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id' : {'session_id' : EEGData}}
        """
        subjects = tqdm(
            self.subjects,
            "Load EEGData",
            unit="sub",
            dynamic_ncols=True,
            disable=not progressbar,
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_data(subject, verbose)
        return data


class EpochsDataset(RawDataset):
    """DPEEG Epochs EEG Dataset.

    Datasets with ``mne.Epochs`` as the base class, such as motor imagery,
    where each subject performs a different imagery task. ``get_raw()`` is used
    to obtain the raw EEG data of each subject, ``get_epochs()`` is used to
    obtain the Epochs of each subject, and ``get_data()`` is used to obtain the
    ``EEGData`` of each subject.
    """

    def __init__(
        self,
        repr: dict,
        subject_list: list[int],
        interval: list[float],
        event_id: dict[str, int],
        subjects: list[int] | None = None,
        tmin: float = 0.0,
        tmax: float | None = None,
        baseline: tuple[int, int] | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        unit_factor: float = 1e6,
    ) -> None:
        super().__init__(
            repr=repr,
            subject_list=subject_list,
            event_id=event_id,
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            resample=resample,
            unit_factor=unit_factor,
        )

        self.interval = interval
        self.baseline = baseline

    def _get_single_subject_epochs(
        self, subject: int, verbose="ERROR"
    ) -> dict[str, Epochs]:
        data = {}
        for session, runs in self._get_single_subject_raw(subject, verbose).items():  # type: ignore
            epochs = []
            for run, raw in runs.items():
                proc = self._epochs_from_raw(raw)
                if proc is None:
                    # if the run did not contain any selected event go to next
                    continue

                epochs.append(proc)
            data[session] = mne.concatenate_epochs(epochs, verbose=verbose)
        return data

    def get_epochs(
        self, progressbar: bool = True, verbose="ERROR"
    ) -> dict[int, dict[str, Epochs]]:
        """Return the epochs correspoonding to a list of subjects.

        The returned data is a dictionary with the following structure:

            data = {'subject_id' : {'session_id' : Epochs}}
        """
        subjects = tqdm(
            self.subjects,
            "Load Epochs",
            unit="sub",
            dynamic_ncols=True,
            disable=not progressbar,
        )

        data = {}
        for subject in subjects:
            data[subject] = self._get_single_subject_epochs(subject, verbose)
        return data

    def _get_single_subject_data(
        self, subject: int, verbose="ERROR"
    ) -> MultiSessEEGData:
        sessions = self._get_single_subject_epochs(subject, verbose)

        data = []
        for session, epochs in sessions.items():
            data.append(self._data_from_epochs(epochs, verbose))
        return MultiSessEEGData(data)

    def _epochs_from_raw(self, raw: Raw, verbose="ERROR") -> Epochs:
        events, event_id = self._events_from_raw(raw, verbose)

        # get interval
        tmin = self.tmin + self.interval[0]
        if self.tmax is None:
            tmax = self.interval[1]
        else:
            tmax = self.tmax + self.interval[0]

        # epoch data
        baseline = self.baseline
        if baseline is not None:
            baseline = (
                baseline[0] + self.interval[0],
                baseline[1] + self.interval[1],
            )
            bmin = baseline[0] if baseline[0] < tmin else tmin
            bmax = baseline[1] if baseline[1] > tmax else tmax
        else:
            bmin = tmin
            bmax = tmax
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=bmin,
            tmax=bmax,
            proj=False,
            baseline=baseline,
            preload=True,
            event_repeated="drop",
            verbose=verbose,
        )
        if bmin < tmin or bmax > tmax:
            epochs.crop(tmin=tmin, tmax=tmax)
        return epochs.crop(include_tmax=False)

    def _data_from_epochs(self, epochs: Epochs, verbose="ERROR") -> EEGData:
        if self.picks is None:
            picks = mne.pick_types(epochs.info, eeg=True, stim=False)
        else:
            picks = mne.pick_channels(
                epochs.info["ch_names"], include=self.picks, ordered=True
            )
        epochs.pick(picks, verbose=verbose)

        if self.resample is not None:
            epochs.resample(self.resample, verbose=verbose)

        edata = self.unit_factor * epochs.get_data(copy=False)
        label = epochs.events[:, -1]
        return EEGData(edata, label)

    def _events_from_raw(self, raw: Raw, verbose="ERROR"):
        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if len(stim_channels) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=verbose)
            event_id = self.event_id
        else:
            events, event_id = mne.events_from_annotations(
                raw, event_id=self.event_id, verbose=verbose  # type: ignore
            )
            # offset = int(self.interval[0] * raw.info["sfreq"])
            # events[:, 0] -= offset  # return the original events onset

        return events, event_id
