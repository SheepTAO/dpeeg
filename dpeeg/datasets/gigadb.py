import numpy as np
from scipy.io import loadmat
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne import create_info

from .base import _EEGDataset, DATA_PATH
from .download import data_dl
from ..utils import get_init_args
from ..tools.docs import fill_doc


URL = "ftp://parrot.genomics.cn/gigadb/pub/"


@fill_doc
class OpenBMI_MI(_EEGDataset):
    """BMI/OpenBMI dataset for MI.

    .. admonition:: Dataset summary

        ====== ====== ======= ===== ======== ======
        Subj   Chan   Time    Cls   Freq     Sess
        ====== ====== ======= ===== ======== ======
        54     62     4.0 s   2     1000 Hz  2
        ====== ====== ======= ===== ======== ======

    This data set consists of EEG data from 54 subjects of a study published in
    [1]_. EEG signals were recorded with a sampling rate of 1,000 Hz and
    collected with 62 Ag/AgCl electrodes. The EEG amplifier used in the
    experiment was a BrainAmp (Brain Products; Munich, Germany). The channels
    were nasion-referenced and grounded to electrode AFz. Additionally, an EMG
    electrode recorded from each flexor digitorum profundus muscle with the
    olecranon used as reference. The impedances of the EEG electrodes were
    maintained below 10 k during the entire experiment.

    MI paradigm The MI paradigm was designed following a well-established
    system protocol. For all blocks, the first 3 s of each trial began with a
    black fixation cross that appeared at the center of the monitor to prepare
    subjects for the MI task. Afterwards, the subject performed the imagery
    task of grasping with the appropriate hand for 4 s when the right or left
    arrow appeared as a visual cue. After each task, the screen remained blank
    for 6 s (± 1.5 s). The experiment consisted of training and test phases;
    each phase had 100 trials with balanced right and left hand imagery tasks.
    During the online test phase, the fixation cross appeared at the center of
    the monitor and moved right or left, according to the real-time classifier
    output of the EEG signal.

    References
    ----------
    .. [1] Lee, M. H., Kwon, O. Y., Kim, Y. J., Kim, H. K., Lee, Y. E.,
        Williamson, J., … Lee, S. W. (2019). EEG dataset and OpenBMI toolbox
        for three BCI paradigms: An investigation into BCI illiteracy.
        GigaScience, 8(5), 1-16. https://doi.org/10.1093/gigascience/giz002

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
            subject_list=list(range(1, 55)),
            interval=[0.0, 4.0],
            event_id={"right_hand": 1, "left_hand": 2},
            subjects=subjects,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample=resample,
        )
        self._data_url = f"{URL}10.5524/100001_101000/100542/"
        self._data_path = DATA_PATH / "openbmi_mi"

    def _load_run(self, data, verbose):
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

        raw = RawArray(data=raw_data, info=info, verbose=verbose)
        raw.set_montage(make_standard_montage("standard_1005"))

        return raw

    def _load_sess(self, filename, verbose):
        mat = loadmat(filename)
        train = self._load_run(mat["EEG_MI_train"][0, 0], verbose)
        test = self._load_run(mat["EEG_MI_test"][0, 0], verbose)
        return {"train": train, "test": test}

    def _get_single_subject_raw(self, subject: int, verbose="ERROR"):
        sessions = {}
        for r in [1, 2]:
            filename = data_dl(
                f"{self._data_url}session{r}/s{subject}/sess{r:02d}_subj{subject:02d}_EEG_MI.mat",
                self._data_path,
                force_update=False,
            )
            sessions[f"session_{r}"] = self._load_sess(filename, verbose)

        return sessions
