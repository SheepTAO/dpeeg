import numpy as np
from scipy.io import loadmat
from numpy import ndarray
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne import create_info

from .base import EpochsDataset, DATA_PATH
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


URL = "http://bnci-horizon-2020.eu/database/data-sets/"


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


def _convert_run(run, ch_names, ch_types, verbose):
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
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw, event_id


def _convert_mi(filename, ch_names, ch_types, verbose):
    runs = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    if isinstance(data["data"], ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        raw, evd = _convert_run(run, ch_names, ch_types, verbose)
        if raw is None:
            continue
        runs.append(raw)
        event_id.update(evd)  # type: ignore
        # change labels to match rest
    standardize_keys(event_id)
    return runs, event_id


@fill_doc
class BCICIV2A(EpochsDataset):
    """Dataset IIA from BCI Competition IV.

    .. admonition:: Dataset summary

        ====== ====== ====== ===== ======= ======
        Subj   Chan   Time   Cls   Freq    Sess
        ====== ====== ====== ===== ======= ======
        9      22     4 s    4     250 Hz  2
        ====== ====== ====== ===== ======= ======

    This data set consists of EEG data from 9 subjects of a study published in
    [1]_.  The cue-based BCI paradigm consisted of four different motor imagery
    tasks, namely the imagination of movement of the left hand (class 1), right
    hand (class 2), both feet (class 3), and tongue (class 4). Two sessions on
    different days were recorded for each subject.  Each session is comprised
    of 6 runs separated by short breaks. One run consists of 48 trials (12 for
    each of the four possible classes), yielding a total of 288 trials per
    session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left. All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 μV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------
    .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
        Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G. and
        Nolte, G., 2012. Review of the BCI competition IV.
        Frontiers in neuroscience, 6, p.55.

    Parameters
    ----------
    %(subjects)s
    %(epochs_tmin_tmax)s
    %(baseline)s
    %(picks)s
    %(resample)s
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
        rename: str | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            subject_list=list(range(1, 10)),
            interval=[2.0, 6.0],
            event_id={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
        )
        # fmt: off
        self._ch_names = [
            "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
            "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
            "EOG1", "EOG2", "EOG3",
        ]
        # fmt: on
        self._ch_types = ["eeg"] * 22 + ["eog"] * 3
        self._data_url = f"{URL}001-2014/"
        self._data_path = DATA_PATH / "bciciv2a"

    def _get_single_subject_raw(self, subject: int, verbose="ERROR"):
        sessions = {}
        for i, r in enumerate(["T", "E"], start=1):
            filename = data_dl(
                f"{self._data_url}A{subject:02d}{r}.mat",
                self._data_path,
                force_update=False,
            )
            runs, _ = _convert_mi(
                filename, self._ch_names, self._ch_types, verbose=verbose
            )
            sessions[f"session_{i}"] = {
                f"run_{ii}": run for ii, run in enumerate(runs, start=1)
            }
        return sessions


@fill_doc
class BCICIV2B(EpochsDataset):
    """Dataset IIB from BCI Competition IV.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        9      3      4.5 s   2     250 Hz   5
        ====== ====== ======= ===== ======== ======


    This data set consists of EEG data from 9 subjects of a study published in
    [1]_. The subjects were right-handed, had normal or corrected-to-normal
    vision and were paid for participating in the experiments.
    All volunteers were sitting in an armchair, watching a flat screen monitor
    placed approximately 1 m away at eye level. For each subject 5 sessions
    are provided, whereby the first two sessions contain training data without
    feedback (screening), and the last three sessions were recorded with
    feedback.

    Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
    frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
    and a notch filter at 50 Hz was enabled.  The placement of the three
    bipolar recordings (large or small distances, more anterior or posterior)
    were slightly different for each subject (for more details see [1]).
    The electrode position Fz served as EEG ground. In addition to the EEG
    channels, the electrooculogram (EOG) was recorded with three monopolar
    electrodes.

    The cue-based screening paradigm consisted of two classes,
    namely the motor imagery (MI) of left hand (class 1) and right hand
    (class 2).
    Each subject participated in two screening sessions without feedback
    recorded on two different days within two weeks.
    Each session consisted of six runs with ten trials each and two classes of
    imagery.  This resulted in 20 trials per run and 120 trials per session.
    Data of 120 repetitions of each MI class were available for each person in
    total.  Prior to the first motor im- agery training the subject executed
    and imagined different movements for each body part and selected the one
    which they could imagine best (e. g., squeezing a ball or pulling a brake).

    Each trial started with a fixation cross and an additional short acoustic
    warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
    for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
    hand movement over a period of 4 seconds.  Each trial was followed by a
    short break of at least 1.5 seconds.  A randomized time of up to 1 second
    was added to the break to avoid adaptation

    For the three online feedback sessions four runs with smiley feedback
    were recorded, whereby each run consisted of twenty trials for each type of
    motor imagery.  At the beginning of each trial (second 0) the feedback (a
    gray smiley) was centered on the screen.  At second 2, a short warning beep
    (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
    second 7.5 the screen went blank and a random interval between 1.0 and 2.0
    seconds was added to the trial.

    References
    ----------
    .. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
        G. Pfurtscheller. Brain-computer communication: motivation, aim, and
        impact of exploring a virtual apartment. IEEE Transactions on Neural
        Systems and Rehabilitation Engineering 15, 473-482, 2007.

    Parameters
    ----------
    %(subjects)s
    %(epochs_tmin_tmax)s
    %(baseline)s
    %(picks)s
    %(resample)s
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
        rename: str | None = None,
    ) -> None:
        super().__init__(
            repr=get_init_args(self, locals(), rename=rename, ret_dict=True),
            subject_list=list(range(1, 10)),
            interval=[3.0, 7.5],
            event_id={"left_hand": 1, "right_hand": 2},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
        )
        self._ch_names = ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"]
        self._ch_types = ["eeg"] * 3 + ["eog"] * 3
        self._data_url = f"{URL}004-2014/"
        self._data_path = DATA_PATH / "bciciv2b"

    def _get_single_subject_raw(self, subject: int, verbose="ERROR"):
        sessions = []
        for i, r in enumerate(["T", "E"], start=1):
            filename = data_dl(
                f"{self._data_url}B{subject:02d}{r}.mat",
                self._data_path,
                force_update=False,
            )
            runs, _ = _convert_mi(
                filename, self._ch_names, self._ch_types, verbose=verbose
            )
            sessions.extend(runs)

        sessions = {
            f"session_{ii}": {"run_1": run} for ii, run in enumerate(sessions, start=1)
        }
        return sessions
