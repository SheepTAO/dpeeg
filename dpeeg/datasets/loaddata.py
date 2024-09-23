# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from numpy import ndarray
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne import create_info

from .download import data_dl
from ..utils import DPEEG_DIR


GIGADB_URL = "ftp://parrot.genomics.cn/gigadb/pub/"
PHYSIONET_URL = "https://physionet.org/files/"
BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"


def load_data(
    subject,
    dataset: str,
    path: str | None = None,
    force_update: bool = False,
):
    """Return the data of a single subject.

    The returned data is a dictionary with the folowing structure:

        data = {'session_id':
                    {'run_id': raw}
                }

    Parameters
    ----------
    subject : int
        The subject to load.
    dataset : string
        The dataset name.
    path : str, optional
        Location of where to look for the data storing location. Default is
        `~/dpeeg/dataset`. If the dataset is not found under the given path,
        the data will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.

    Returns
    -------
    raws : dict
        dict containing the raw data.
    """
    dataset_list = {
        "bciciv2a": {"url": BNCI_URL, "load": _load_bciciv2a},
        "bciciv2b": {"url": BNCI_URL, "load": _load_bciciv2b},
        "openbmi_mi": {
            "url": f"{GIGADB_URL}10.5524/100001_101000/100542/",
            "load": _load_openbmi_mi,
        },
    }

    if dataset not in dataset_list.keys():
        raise ValueError(
            f"Dataset {dataset} is not available. "
            f"Valid dataset are {', '.join(dataset_list.keys())}."
        )

    dl_path = Path(DPEEG_DIR, "datasets") if path is None else Path(path)
    return dataset_list[dataset]["load"](
        subject, dl_path / dataset, dataset_list[dataset]["url"], force_update
    )


def standardize_keys(d):
    master_list = [
        ["both feet", "feet"],
        ["left hand", "left_hand"],
        ["right hand", "right_hand"],
        ["FEET", "feet"],
        ["HAND", "right_hand"],
        ["NAV", "navigation"],
        ["SUB", "subtraction"],
        ["WORD", "word_ass"],
    ]
    for old, new in master_list:
        if old in d.keys():
            d[new] = d.pop(old)


def _convert_run(run, ch_names, ch_types):
    """Convert one run to raw."""
    event_id = {}
    n_chan = run.X.shape[1]
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    if not ch_names:
        ch_names = [f"EEG{ch}" for ch in range(1, n_chan + 1)]
        montage = None

    if not ch_types:
        ch_types = ["eeg"] * n_chan

    trigger = np.zeros((len(eeg_data), 1))
    # some runs does not contains trials i.e baseline runs
    if len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y
    else:
        return None, None

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ["stim"]
    ch_types = ch_types + ["stim"]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=False)
    raw.set_montage(montage)
    return raw, event_id


def _convert_mi(filename, ch_names, ch_types):
    runs = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    if isinstance(data["data"], ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        raw, evd = _convert_run(run, ch_names, ch_types)
        if raw is None:
            continue
        runs.append(raw)
        event_id.update(evd)  # type: ignore
        # change labels to match rest
    standardize_keys(event_id)
    return runs, event_id


def _load_bciciv2a(
    subject: int,
    path: str,
    base_url: str,
    force_update: bool = False,
):
    """Load data set 2a of the BCI Competition IV."""
    if (subject < 1) or (subject > 9):
        raise ValueError(f"Subject must be between 1 and 9. Got {subject}.")

    # fmt: off
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
        "EOG1", "EOG2", "EOG3",
    ]
    # fmt: on
    ch_types = ["eeg"] * 22 + ["eog"] * 3

    sessions = {}
    for i, r in enumerate(["T", "E"], start=1):
        url = f"{base_url}001-2014/A{subject:02d}{r}.mat"
        filename = data_dl(url, path, force_update)
        runs, _ = _convert_mi(filename, ch_names, ch_types)
        sessions[f"session_{i}"] = {
            f"run_{ii}": run for ii, run in enumerate(runs, start=1)
        }
    return sessions


def _load_bciciv2b(
    subject: int,
    path: str,
    base_url: str,
    force_updata: bool = False,
):
    """Load data set 2b of the BCI Competition IV."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    ch_names = ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"]
    ch_types = ["eeg"] * 3 + ["eog"] * 3

    sessions = []
    for i, r in enumerate(["T", "E"], start=1):
        url = f"{base_url}004-2014/B{subject:02d}{r}.mat"
        filename = data_dl(url, path, force_updata)
        runs, _ = _convert_mi(filename, ch_names, ch_types)
        sessions.extend(runs)

    sessions = {
        f"session_{ii}": {"run_1": run} for ii, run in enumerate(sessions, start=1)
    }
    return sessions


def _load_openbmi_mi_run(data):
    sfreq = data["fs"].item()

    eeg_ch_names = [np.squeeze(c).item() for c in np.ravel(data["chan"])]
    eeg_ch_types = ["eeg"] * len(eeg_ch_names)
    emg_ch_names = [np.squeeze(c).item() for c in np.ravel(data["EMG_index"])]
    emg_ch_types = ["emg"] * len(emg_ch_names)
    raw_ch_names = eeg_ch_names + emg_ch_names + ["STI 014"]
    raw_ch_types = eeg_ch_types + emg_ch_types + ["stim"]
    info = create_info(raw_ch_names, sfreq, raw_ch_types)  # type: ignore

    eeg = data["x"].transpose(1, 0) * 1e-6  # to load the signal in Volts
    emg = data["EMG"].transpose(1, 0) * 1e-6  # to load the signal in Volts
    event_times_in_samples = data["t"].squeeze()
    event_id = data["y_dec"].squeeze()
    stim = np.zeros(len(data["x"]))
    for i_sample, id_class in zip(event_times_in_samples, event_id):
        stim[i_sample] += id_class
    raw_data = np.concatenate([eeg, emg, stim[None, :]])

    raw = RawArray(data=raw_data, info=info, verbose=False)
    raw.set_montage(make_standard_montage("standard_1005"))

    return raw


def _load_openbmi_mi_sess(filename):
    mat = loadmat(filename)
    train = _load_openbmi_mi_run(mat["EEG_MI_train"][0, 0])
    test = _load_openbmi_mi_run(mat["EEG_MI_test"][0, 0])
    return {"train": train, "test": test}


def _load_openbmi_mi(
    subject: int,
    path: str,
    base_url: str,
    force_update: bool = False,
):
    """Load data set MI of the OpenBMI."""
    if (subject < 1) or (subject > 54):
        raise ValueError("Subject must be between 1 and 54. Got %d." % subject)

    sessions = {}
    for r in [1, 2]:
        url = (
            f"{base_url}session{r}/s{subject}/sess{r:02d}_"
            f"subj{subject:02d}_EEG_MI.mat"
        )
        filename = data_dl(url, path, force_update)
        sessions[f"session_{r}"] = _load_openbmi_mi_sess(filename)

    return sessions
