import numpy as np
from mne.io import read_raw_edf
from mne.channels import make_standard_montage

from .base import EpochsDataset, DATA_PATH
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


URL = "https://physionet.org/files/"


@fill_doc
class PhysioNet_MI(EpochsDataset):
    """Physionet Motor Imagery dataset.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        109    64     3.0 s   4     160 Hz   1
        ====== ====== ======= ===== ======== ======

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers [2]_.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org) [1]_.
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
    The subject opens and closes the corresponding fist until the target
    disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
    The subject imagines opening and closing the corresponding fist until
    the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
    The subject opens and closes either both fists (if the target is on top)
    or both feet (if the target is on the bottom) until the target
    disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
    The subject imagines opening and closing either both fists
    (if the target is on top) or both feet (if the target is on the bottom)
    until the target disappears. Then the subject relaxes.

    references
    ----------
    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
        Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
        interface (BCI) system. IEEE Transactions on biomedical engineering,
        51(6), pp.1034-1043.
    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
        P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
        H.E. and PhysioBank, P., PhysioNet: components of a new research
        resource for complex physiologic signals Circulation 2000 Volume
        101 Issue 23 pp. E215-E220.

    Parameters
    ----------
    %(subjects)s
    %(epochs_tmin_tmax)s
    %(baseline)s
    %(picks)s
    %(resample)s
    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.
    executed: bool (default False)
        if True, return runs corresponding to motor execution.
    %(rename)s
    """

    def __init__(
        self,
        subjects: list[int] | None = None,
        tmin: float = 0,
        tmax: float | None = None,
        baseline: tuple[int, int] | None = None,
        picks: list[str] | None = None,
        resample: float | None = None,
        imagined: bool = True,
        executed: bool = False,
        rename: str | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            subject_list=list(range(1, 110)),
            interval=[0.0, 3.0],
            event_id={
                "rest": 1,
                "left_hand": 2,
                "right_hand": 3,
                "hands": 4,
                "feet": 5,
            },
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
        )
        self._data_path = DATA_PATH / "physionet_mi"
        self._data_url = f"{URL}eegmmidb/1.0.0/"

        # fmt: off
        self._renames = {
            "AFZ": "AFz", "PZ": "Pz", "FPZ": "Fpz", "FCZ": "FCz", "FP1": "Fp1", 
            "CZ": "Cz", "OZ": "Oz", "POZ": "POz", "IZ": "Iz", "CPZ": "CPz", 
            "FP2": "Fp2", "FZ": "Fz",
        }
        # fmt: on
        self._montage = make_standard_montage("standard_1005")

        self._feet_runs = []
        self._hand_runs = []
        if imagined:
            self._feet_runs += [6, 10, 14]
            self._hand_runs += [4, 8, 12]
        if executed:
            self._feet_runs += [5, 9, 13]
            self._hand_runs += [3, 7, 11]

    def _load_run(self, subject, run, verbose):
        filename = data_dl(
            f"{self._data_url}S{subject:03d}/S{subject:03d}R{run:02d}.edf",
            self._data_path,
            force_update=False,
        )
        raw = read_raw_edf(filename, preload=True, verbose=verbose)
        raw.rename_channels(lambda x: x.strip("."))
        raw.rename_channels(lambda x: x.upper())
        raw.rename_channels(self._renames)
        raw.set_montage(self._montage)
        return raw

    def _get_single_subject_raw(self, subject: int, verbose="ERROR"):
        session = {}

        # hand runs
        idx = 1
        for run in self._hand_runs:
            raw = self._load_run(subject, run, verbose)
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "left_hand"
            stim[stim == "T2"] = "right_hand"
            raw.annotations.description = stim
            session[f"run_{idx}"] = raw
            idx += 1

        # feet runs
        for run in self._feet_runs:
            raw = self._load_run(subject, run, verbose)
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "hands"
            stim[stim == "T2"] = "feet"
            raw.annotations.description = stim
            session[f"run_{idx}"] = raw
            idx += 1

        return {"session_1": session}
